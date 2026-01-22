from __future__ import division
import re
import stat
from .helpers import (
class FeatureCommand(ImportCommand):

    def __init__(self, feature_name, value=None, lineno=0):
        ImportCommand.__init__(self, b'feature')
        self.feature_name = feature_name
        self.value = value
        self.lineno = lineno

    def __bytes__(self):
        if self.value is None:
            value_text = b''
        else:
            value_text = b'=' + self.value
        return b'feature ' + self.feature_name + value_text