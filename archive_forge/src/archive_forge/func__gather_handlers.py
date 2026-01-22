from __future__ import annotations
import getopt
import inspect
import os
import sys
import textwrap
from os import path
from typing import Any, Dict, Optional, cast
from twisted.python import reflect, util
def _gather_handlers(self):
    """
        Gather up options with their own handler methods.

        This returns a tuple of many values.  Amongst those values is a
        synonyms dictionary, mapping all of the possible aliases (C{str})
        for an option to the longest spelling of that option's name
        C({str}).

        Another element is a dispatch dictionary, mapping each user-facing
        option name (with - substituted for _) to a callable to handle that
        option.
        """
    longOpt, shortOpt = ([], '')
    docs, settings, synonyms, dispatch = ({}, {}, {}, {})
    dct = {}
    reflect.addMethodNamesToDict(self.__class__, dct, 'opt_')
    for name in dct.keys():
        method = getattr(self, 'opt_' + name)
        takesArg = not flagFunction(method, name)
        prettyName = name.replace('_', '-')
        doc = getattr(method, '__doc__', None)
        if doc:
            docs[prettyName] = doc
        else:
            docs[prettyName] = self.docs.get(prettyName)
        synonyms[prettyName] = prettyName
        if takesArg:
            fn = lambda name, value, m=method: m(value)
        else:
            fn = lambda name, value=None, m=method: m()
        dispatch[prettyName] = fn
        if len(name) == 1:
            shortOpt = shortOpt + name
            if takesArg:
                shortOpt = shortOpt + ':'
        else:
            if takesArg:
                prettyName = prettyName + '='
            longOpt.append(prettyName)
    reverse_dct = {}
    for name in dct.keys():
        method = getattr(self, 'opt_' + name)
        if method not in reverse_dct:
            reverse_dct[method] = []
        reverse_dct[method].append(name.replace('_', '-'))
    for method, names in reverse_dct.items():
        if len(names) < 2:
            continue
        longest = max(names, key=len)
        for name in names:
            synonyms[name] = longest
    return (longOpt, shortOpt, docs, settings, synonyms, dispatch)