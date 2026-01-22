import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
def _load_history_from_file(self):
    if os.path.isfile(self._history_file_path):
        try:
            with open(self._history_file_path, 'rt') as history_file:
                commands = history_file.readlines()
            self._commands = [command.strip() for command in commands if command.strip()]
            if len(self._commands) > self._limit:
                self._commands = self._commands[-self._limit:]
                with open(self._history_file_path, 'wt') as history_file:
                    for command in self._commands:
                        history_file.write(command + '\n')
        except IOError:
            print('WARNING: writing history file failed.')