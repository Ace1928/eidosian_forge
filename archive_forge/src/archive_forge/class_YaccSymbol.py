import re
import types
import sys
import os.path
import inspect
import base64
import warnings
class YaccSymbol:

    def __str__(self):
        return self.type

    def __repr__(self):
        return str(self)