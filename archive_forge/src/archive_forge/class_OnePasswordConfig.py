from __future__ import (absolute_import, division, print_function)
import os
class OnePasswordConfig(object):
    _config_file_paths = ('~/.op/config', '~/.config/op/config', '~/.config/.op/config')

    def __init__(self):
        self._config_file_path = ''

    @property
    def config_file_path(self):
        if self._config_file_path:
            return self._config_file_path
        for path in self._config_file_paths:
            realpath = os.path.expanduser(path)
            if os.path.exists(realpath):
                self._config_file_path = realpath
                return self._config_file_path