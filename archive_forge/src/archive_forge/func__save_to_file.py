import collections
import json
import os
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.platform import gfile
def _save_to_file(self):
    try:
        with gfile.Open(self._config_file_path, 'w') as config_file:
            json.dump(self._config, config_file)
    except IOError:
        pass