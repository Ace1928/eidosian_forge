from __future__ import print_function
import re
from sqlparse import tokens as T
from sqlparse.compat import string_types, text_type, unicode_compatible
from sqlparse.utils import imt, remove_quotes
def token_first(self, skip_ws=True, skip_cm=False):
    """Returns the first child token.

        If *skip_ws* is ``True`` (the default), whitespace
        tokens are ignored.

        if *skip_cm* is ``True`` (default: ``False``), comments are
        ignored too.
        """
    funcs = lambda tk: not (skip_ws and tk.is_whitespace or (skip_cm and imt(tk, t=T.Comment, i=Comment)))
    return self._token_matching(funcs)[1]