import gyp.common
import json
import os
import posixpath
def _ExtractSources(target, target_dict, toplevel_dir):
    base_path = posixpath.dirname(_ToLocalPath(toplevel_dir, _ToGypPath(target)))
    base_path_components = base_path.split('/')
    if len(base_path):
        base_path += '/'
    if debug:
        print('ExtractSources', target, base_path)
    results = []
    if 'sources' in target_dict:
        _AddSources(target_dict['sources'], base_path, base_path_components, results)
    if 'actions' in target_dict:
        for action in target_dict['actions']:
            _ExtractSourcesFromAction(action, base_path, base_path_components, results)
    if 'rules' in target_dict:
        for rule in target_dict['rules']:
            _ExtractSourcesFromAction(rule, base_path, base_path_components, results)
    return results