from __future__ import annotations
import getopt
import inspect
import os
import sys
import textwrap
from os import path
from typing import Any, Dict, Optional, cast
from twisted.python import reflect, util
def _gather_flags(self):
    """
        Gather up boolean (flag) options.
        """
    longOpt, shortOpt = ([], '')
    docs, settings, synonyms, dispatch = ({}, {}, {}, {})
    flags = []
    reflect.accumulateClassList(self.__class__, 'optFlags', flags)
    for flag in flags:
        long, short, doc = util.padTo(3, flag)
        if not long:
            raise ValueError('A flag cannot be without a name.')
        docs[long] = doc
        settings[long] = 0
        if short:
            shortOpt = shortOpt + short
            synonyms[short] = long
        longOpt.append(long)
        synonyms[long] = long
        dispatch[long] = self._generic_flag
    return (longOpt, shortOpt, docs, settings, synonyms, dispatch)