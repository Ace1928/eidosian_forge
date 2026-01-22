import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
def _add_command_to_history_file(self, command):
    try:
        with open(self._history_file_path, 'at') as history_file:
            history_file.write(command + '\n')
    except IOError:
        pass