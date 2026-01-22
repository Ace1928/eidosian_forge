from __future__ import absolute_import
from __future__ import unicode_literals
import logging
import os
from googlecloudsdk.third_party.appengine.api import appinfo
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.ext import builtins
def _ConvertBuiltinsToIncludes(included_from, app_include, state, runtime):
    includes_list = []
    if app_include.builtins:
        builtins_list = appinfo.BuiltinHandler.ListToTuples(app_include.builtins)
        for builtin_name, on_or_off in builtins_list:
            if not on_or_off:
                continue
            yaml_path = builtins.get_yaml_path(builtin_name, runtime)
            if on_or_off == 'on':
                includes_list.append(yaml_path)
            elif on_or_off == 'off':
                if yaml_path in state.includes:
                    logging.warning('%s already included by %s but later disabled by %s', yaml_path, state.includes[yaml_path], included_from)
                state.excludes[yaml_path] = included_from
            else:
                logging.error('Invalid state for AppInclude object loaded from %s; builtins directive "%s: %s" ignored.', included_from, builtin_name, on_or_off)
    return includes_list