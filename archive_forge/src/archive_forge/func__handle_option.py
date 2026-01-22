from __future__ import annotations
from collections.abc import MutableMapping
from collections import ChainMap as _ChainMap
import contextlib
from dataclasses import dataclass, field
import functools
from .compat import io
import itertools
import os
import re
import sys
from typing import Iterable
def _handle_option(self, st, line, fpname):
    st.indent_level = st.cur_indent_level
    mo = self._optcre.match(line.clean)
    if not mo:
        st.errors.append(ParsingError(fpname, st.lineno, line))
        return
    st.optname, vi, optval = mo.group('option', 'vi', 'value')
    if not st.optname:
        st.errors.append(ParsingError(fpname, st.lineno, line))
    st.optname = self.optionxform(st.optname.rstrip())
    if self._strict and (st.sectname, st.optname) in st.elements_added:
        raise DuplicateOptionError(st.sectname, st.optname, fpname, st.lineno)
    st.elements_added.add((st.sectname, st.optname))
    if optval is not None:
        optval = optval.strip()
        st.cursect[st.optname] = [optval]
    else:
        st.cursect[st.optname] = None