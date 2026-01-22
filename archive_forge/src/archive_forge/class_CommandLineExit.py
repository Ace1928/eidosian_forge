import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
class CommandLineExit(Exception):

    def __init__(self, exit_token=None):
        Exception.__init__(self)
        self._exit_token = exit_token

    @property
    def exit_token(self):
        return self._exit_token