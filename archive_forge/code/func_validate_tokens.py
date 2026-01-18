import re
import sys
import types
import copy
import os
import inspect
def validate_tokens(self):
    terminals = {}
    for n in self.tokens:
        if not _is_identifier.match(n):
            self.log.error("Bad token name '%s'", n)
            self.error = True
        if n in terminals:
            self.log.warning("Token '%s' multiply defined", n)
        terminals[n] = 1