import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class RecursiveGrammarException(Exception):
    """exception thrown by C{validate()} if the grammar could be improperly recursive"""

    def __init__(self, parseElementList):
        self.parseElementTrace = parseElementList

    def __str__(self):
        return 'RecursiveGrammarException: %s' % self.parseElementTrace