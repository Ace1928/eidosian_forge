import gyp.common
import json
import os
import posixpath
def _ExtractSourcesFromAction(action, base_path, base_path_components, results):
    if 'inputs' in action:
        _AddSources(action['inputs'], base_path, base_path_components, results)