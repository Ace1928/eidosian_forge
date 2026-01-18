from passlib.utils.compat import JYTHON
from binascii import b2a_base64, a2b_base64, Error as _BinAsciiError
from base64 import b64encode, b64decode
from codecs import lookup as _lookup_codec
from functools import update_wrapper
import itertools
import inspect
import logging; log = logging.getLogger(__name__)
import math
import os
import sys
import random
import re
import time
import timeit
import types
from warnings import warn
from passlib.utils.binary import (
from passlib.utils.decor import (
from passlib.exc import ExpectedStringError, ExpectedTypeError
from passlib.utils.compat import (add_doc, join_bytes, join_byte_values,
from passlib.exc import MissingBackendError
def update_mixin_classes(target, add=None, remove=None, append=False, before=None, after=None, dryrun=False):
    """
    helper to update mixin classes installed in target class.

    :param target:
        target class whose bases will be modified.

    :param add:
        class / classes to install into target's base class list.

    :param remove:
        class / classes to remove from target's base class list.

    :param append:
        by default, prepends mixins to front of list.
        if True, appends to end of list instead.

    :param after:
        optionally make sure all mixins are inserted after
        this class / classes.

    :param before:
        optionally make sure all mixins are inserted before
        this class / classes.

    :param dryrun:
        optionally perform all calculations / raise errors,
        but don't actually modify the class.
    """
    if isinstance(add, type):
        add = [add]
    bases = list(target.__bases__)
    if remove:
        if isinstance(remove, type):
            remove = [remove]
        for mixin in remove:
            if add and mixin in add:
                continue
            if mixin in bases:
                bases.remove(mixin)
    if add:
        for mixin in add:
            if any((issubclass(base, mixin) for base in bases)):
                continue
            if append:
                for idx, base in enumerate(bases):
                    if issubclass(mixin, base):
                        break
                    if before and issubclass(base, before):
                        break
                else:
                    idx = len(bases)
            elif after:
                for end_idx, base in enumerate(reversed(bases)):
                    if issubclass(base, after):
                        idx = len(bases) - end_idx
                        assert bases[idx - 1] == base
                        break
                else:
                    idx = 0
            else:
                idx = 0
            bases.insert(idx, mixin)
    if not dryrun:
        target.__bases__ = tuple(bases)