from __future__ import annotations
import getopt
import inspect
import os
import sys
import textwrap
from os import path
from typing import Any, Dict, Optional, cast
from twisted.python import reflect, util
def _gather_parameters(self):
    """
        Gather options which take a value.
        """
    longOpt, shortOpt = ([], '')
    docs, settings, synonyms, dispatch = ({}, {}, {}, {})
    parameters = []
    reflect.accumulateClassList(self.__class__, 'optParameters', parameters)
    synonyms = {}
    for parameter in parameters:
        long, short, default, doc, paramType = util.padTo(5, parameter)
        if not long:
            raise ValueError('A parameter cannot be without a name.')
        docs[long] = doc
        settings[long] = default
        if short:
            shortOpt = shortOpt + short + ':'
            synonyms[short] = long
        longOpt.append(long + '=')
        synonyms[long] = long
        if paramType is not None:
            dispatch[long] = CoerceParameter(self, paramType)
        else:
            dispatch[long] = CoerceParameter(self, str)
    return (longOpt, shortOpt, docs, settings, synonyms, dispatch)