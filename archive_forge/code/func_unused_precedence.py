import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def unused_precedence(self):
    unused = []
    for termname in self.Precedence:
        if not (termname in self.Terminals or termname in self.UsedPrecedence):
            unused.append((termname, self.Precedence[termname][0]))
    return unused