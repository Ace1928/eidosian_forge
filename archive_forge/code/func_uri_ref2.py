from __future__ import annotations
import codecs
import os
import re
import sys
import typing
from decimal import Decimal
from typing import (
from uuid import uuid4
from rdflib.compat import long_type
from rdflib.exceptions import ParserError
from rdflib.graph import ConjunctiveGraph, Graph, QuotedGraph
from rdflib.term import (
from rdflib.parser import Parser
def uri_ref2(self, argstr: str, i: int, res: MutableSequence[Any]) -> int:
    """Generate uri from n3 representation.

        Note that the RDF convention of directly concatenating
        NS and local name is now used though I prefer inserting a '#'
        to make the namesapces look more like what XML folks expect.
        """
    qn: typing.List[Any] = []
    j = self.qname(argstr, i, qn)
    if j >= 0:
        pfx, ln = qn[0]
        if pfx is None:
            assert 0, 'not used?'
            ns = self._baseURI + ADDED_HASH
        else:
            try:
                ns = self._bindings[pfx]
            except KeyError:
                if pfx == '_':
                    res.append(self.anonymousNode(ln))
                    return j
                if not self.turtle and pfx == '':
                    ns = join(self._baseURI or '', '#')
                else:
                    self.BadSyntax(argstr, i, 'Prefix "%s:" not bound' % pfx)
        symb = self._store.newSymbol(ns + ln)
        res.append(self._variables.get(symb, symb))
        return j
    i = self.skipSpace(argstr, i)
    if i < 0:
        return -1
    if argstr[i] == '?':
        v: typing.List[Any] = []
        j = self.variable(argstr, i, v)
        if j > 0:
            res.append(v[0])
            return j
        return -1
    elif argstr[i] == '<':
        st = i + 1
        i = argstr.find('>', st)
        if i >= 0:
            uref = argstr[st:i]
            uref = unicodeEscape8.sub(unicodeExpand, uref)
            uref = unicodeEscape4.sub(unicodeExpand, uref)
            if self._baseURI:
                uref = join(self._baseURI, uref)
            else:
                assert ':' in uref, 'With no base URI, cannot deal with relative URIs'
            if argstr[i - 1] == '#' and (not uref[-1:] == '#'):
                uref += '#'
            symb = self._store.newSymbol(uref)
            res.append(self._variables.get(symb, symb))
            return i + 1
        self.BadSyntax(argstr, j, 'unterminated URI reference')
    elif self.keywordsSet:
        v = []
        j = self.bareWord(argstr, i, v)
        if j < 0:
            return -1
        if v[0] in self.keywords:
            self.BadSyntax(argstr, i, 'Keyword "%s" not allowed here.' % v[0])
        res.append(self._store.newSymbol(self._bindings[''] + v[0]))
        return j
    else:
        return -1