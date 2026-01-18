from __future__ import print_function
import re
from sqlparse import tokens as T
from sqlparse.compat import string_types, text_type, unicode_compatible
from sqlparse.utils import imt, remove_quotes
def within(self, group_cls):
    """Returns ``True`` if this token is within *group_cls*.

        Use this method for example to check if an identifier is within
        a function: ``t.within(sql.Function)``.
        """
    parent = self.parent
    while parent:
        if isinstance(parent, group_cls):
            return True
        parent = parent.parent
    return False