import errno
import operator
import os
import random
import re
import shlex
import socket
import stat
import string
import struct
import sys
import textwrap
import time
import traceback
from functools import reduce
from os import path
from typing import Optional
from twisted.internet import protocol, reactor, task
from twisted.persisted import styles
from twisted.protocols import basic
from twisted.python import _textattributes, log, reflect
def _foldr(f, z, xs):
    """
    Apply a function of two arguments cumulatively to the items of
    a sequence, from right to left, so as to reduce the sequence to
    a single value.

    @type f: C{callable} taking 2 arguments

    @param z: Initial value.

    @param xs: Sequence to reduce.

    @return: Single value resulting from reducing C{xs}.
    """
    return reduce(lambda x, y: f(y, x), reversed(xs), z)