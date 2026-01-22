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
class FakeRepr(object):
    """
            Class representing a non-descript PyObject* value in the inferior
            process for when we don't have a custom scraper, intended to have
            a sane repr().
            """

    def __init__(self, tp_name, address):
        self.tp_name = tp_name
        self.address = address

    def __repr__(self):
        if self.address == 0:
            return '0x0'
        return '<%s at remote 0x%x>' % (self.tp_name, self.address)