import re
import sys
import types
import copy
import os
import inspect
def validate_literals(self):
    try:
        for c in self.literals:
            if not isinstance(c, StringTypes) or len(c) > 1:
                self.log.error('Invalid literal %s. Must be a single character', repr(c))
                self.error = True
    except TypeError:
        self.log.error('Invalid literals specification. literals must be a sequence of characters')
        self.error = True