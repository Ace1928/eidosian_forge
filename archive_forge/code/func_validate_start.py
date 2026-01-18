import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def validate_start(self):
    if self.start is not None:
        if not isinstance(self.start, string_types):
            self.log.error("'start' must be a string")