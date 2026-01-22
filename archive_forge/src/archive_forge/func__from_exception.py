import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import collections
import pprint
import traceback
import types
from datetime import datetime
@classmethod
def _from_exception(cls, pe):
    """
        internal factory method to simplify creating one type of ParseException 
        from another - avoids having __init__ signature conflicts among subclasses
        """
    return cls(pe.pstr, pe.loc, pe.msg, pe.parserElement)