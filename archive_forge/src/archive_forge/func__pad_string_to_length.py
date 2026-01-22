import copy
import re
import numpy as np
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.lib import debug_data
def _pad_string_to_length(string, length):
    return ' ' * (length - len(string)) + string