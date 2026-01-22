from __future__ import absolute_import
from __future__ import unicode_literals
import logging
import os
from googlecloudsdk.third_party.appengine.api import appinfo
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.ext import builtins
def _MergeBuiltinsIncludes(appinfo_path, appyaml):
    """Merges app.yaml files from builtins and includes directives in appyaml.

  Args:
    appinfo_path: the application directory.
    appyaml: the yaml file to obtain builtins and includes directives from.

  Returns:
    A tuple where the first element is the modified appyaml object
    incorporating the referenced yaml files, and the second element is a list
    of the absolute paths of the included files, in no particular order.
  """
    if not appyaml.builtins:
        appyaml.builtins = [appinfo.BuiltinHandler(default='on')]
    elif not appinfo.BuiltinHandler.IsDefined(appyaml.builtins, 'default'):
        appyaml.builtins.append(appinfo.BuiltinHandler(default='on'))
    runtime_for_including = appyaml.runtime
    if runtime_for_including == 'vm':
        runtime_for_including = appyaml.vm_settings.get('vm_runtime', 'python27')
    aggregate_appinclude, include_paths = _ResolveIncludes(appinfo_path, appinfo.AppInclude(builtins=appyaml.builtins, includes=appyaml.includes), os.path.dirname(appinfo_path), runtime_for_including)
    return (appinfo.AppInclude.MergeAppYamlAppInclude(appyaml, aggregate_appinclude), include_paths)