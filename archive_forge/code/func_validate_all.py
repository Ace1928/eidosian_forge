import re
import sys
import types
import copy
import os
import inspect
def validate_all(self):
    self.validate_tokens()
    self.validate_literals()
    self.validate_rules()
    return self.error