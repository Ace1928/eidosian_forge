import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def validate_precedence(self):
    preclist = []
    if self.prec:
        if not isinstance(self.prec, (list, tuple)):
            self.log.error('precedence must be a list or tuple')
            self.error = True
            return
        for level, p in enumerate(self.prec):
            if not isinstance(p, (list, tuple)):
                self.log.error('Bad precedence table')
                self.error = True
                return
            if len(p) < 2:
                self.log.error('Malformed precedence entry %s. Must be (assoc, term, ..., term)', p)
                self.error = True
                return
            assoc = p[0]
            if not isinstance(assoc, string_types):
                self.log.error('precedence associativity must be a string')
                self.error = True
                return
            for term in p[1:]:
                if not isinstance(term, string_types):
                    self.log.error('precedence items must be strings')
                    self.error = True
                    return
                preclist.append((term, assoc, level + 1))
    self.preclist = preclist