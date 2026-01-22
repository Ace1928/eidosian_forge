import sys
import re
import warnings
import io
import collections
import collections.abc
import contextlib
import weakref
from . import ElementPath
fromstring = XML
def _default(self, text):
    prefix = text[:1]
    if prefix == '&':
        try:
            data_handler = self.target.data
        except AttributeError:
            return
        try:
            data_handler(self.entity[text[1:-1]])
        except KeyError:
            from xml.parsers import expat
            err = expat.error('undefined entity %s: line %d, column %d' % (text, self.parser.ErrorLineNumber, self.parser.ErrorColumnNumber))
            err.code = 11
            err.lineno = self.parser.ErrorLineNumber
            err.offset = self.parser.ErrorColumnNumber
            raise err
    elif prefix == '<' and text[:9] == '<!DOCTYPE':
        self._doctype = []
    elif self._doctype is not None:
        if prefix == '>':
            self._doctype = None
            return
        text = text.strip()
        if not text:
            return
        self._doctype.append(text)
        n = len(self._doctype)
        if n > 2:
            type = self._doctype[1]
            if type == 'PUBLIC' and n == 4:
                name, type, pubid, system = self._doctype
                if pubid:
                    pubid = pubid[1:-1]
            elif type == 'SYSTEM' and n == 3:
                name, type, system = self._doctype
                pubid = None
            else:
                return
            if hasattr(self.target, 'doctype'):
                self.target.doctype(name, pubid, system[1:-1])
            elif hasattr(self, 'doctype'):
                warnings.warn('The doctype() method of XMLParser is ignored.  Define doctype() method on the TreeBuilder target.', RuntimeWarning)
            self._doctype = None