from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
class BuiltInMethodProxy(object):

    def __init__(self, ml_name, pyop_m_self):
        self.ml_name = ml_name
        self.pyop_m_self = pyop_m_self

    def __repr__(self):
        return '<built-in method %s of %s object at remote 0x%x>' % (self.ml_name, self.pyop_m_self.safe_tp_name(), self.pyop_m_self.as_address())