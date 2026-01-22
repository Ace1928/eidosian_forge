import collections
import json
import os
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.platform import gfile
def _load_from_file(self):
    try:
        with gfile.Open(self._config_file_path, 'r') as config_file:
            config_dict = json.load(config_file)
            config = collections.OrderedDict()
            for key in sorted(config_dict.keys()):
                config[key] = config_dict[key]
            return config
    except (IOError, ValueError):
        return dict()