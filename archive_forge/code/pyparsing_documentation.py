import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
Extends C{leaveWhitespace} defined in base class, and also invokes C{leaveWhitespace} on
           all contained expressions.