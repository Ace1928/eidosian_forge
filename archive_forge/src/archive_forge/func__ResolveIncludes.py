from __future__ import absolute_import
from __future__ import unicode_literals
import logging
import os
from googlecloudsdk.third_party.appengine.api import appinfo
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.ext import builtins
def _ResolveIncludes(included_from, app_include, basepath, runtime, state=None):
    """Recursively includes all encountered builtins/includes directives.

  This function takes an initial AppInclude object specified as a parameter
  and recursively evaluates every builtins/includes directive in the passed
  in AppInclude and any files they reference.  The sole output of the function
  is an AppInclude object that is the result of merging all encountered
  AppInclude objects.  This must then be merged with the root AppYaml object.

  Args:
    included_from: file that included file was included from.
    app_include: the AppInclude object to resolve.
    basepath: application basepath.
    runtime: name of the runtime.
    state: contains the list of included and excluded files as well as the
           directives of all encountered AppInclude objects.

  Returns:
    A two-element tuple where the first element is the AppInclude object merged
    from following all builtins/includes defined in provided AppInclude object;
    and the second element is a list of the absolute paths of the included
    files, in no particular order.

  Raises:
    IncludeFileNotFound: if file specified in an include statement cannot be
      resolved to an includeable file (result from _ResolvePath is False).
  """

    class RecurseState(object):

        def __init__(self):
            self.includes = {}
            self.excludes = {}
            self.aggregate_appinclude = appinfo.AppInclude()
    if not state:
        state = RecurseState()
    appinfo.AppInclude.MergeAppIncludes(state.aggregate_appinclude, app_include)
    includes_list = _ConvertBuiltinsToIncludes(included_from, app_include, state, runtime)
    includes_list.extend(app_include.includes or [])
    for i in includes_list:
        inc_path = _ResolvePath(included_from, i, basepath)
        if not inc_path:
            raise IncludeFileNotFound('File %s listed in includes directive of %s could not be found.' % (i, included_from))
        if inc_path in state.excludes:
            logging.warning('%s already disabled by %s but later included by %s', inc_path, state.excludes[inc_path], included_from)
        elif not inc_path in state.includes:
            state.includes[inc_path] = included_from
            with open(inc_path, 'r') as yaml_file:
                try:
                    inc_yaml = appinfo.LoadAppInclude(yaml_file)
                    _ResolveIncludes(inc_path, inc_yaml, basepath, runtime, state=state)
                except appinfo_errors.EmptyConfigurationFile:
                    if not os.path.basename(os.path.dirname(inc_path)) == 'default':
                        logging.warning('Nothing to include in %s', inc_path)
    return (state.aggregate_appinclude, list(state.includes.keys()))