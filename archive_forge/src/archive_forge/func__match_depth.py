import os
import re
import sys
from codecs import BOM_UTF8, BOM_UTF16, BOM_UTF16_BE, BOM_UTF16_LE
import six
from ._version import __version__
def _match_depth(self, sect, depth):
    """
        Given a section and a depth level, walk back through the sections
        parents to see if the depth level matches a previous section.
        
        Return a reference to the right section,
        or raise a SyntaxError.
        """
    while depth < sect.depth:
        if sect is sect.parent:
            raise SyntaxError()
        sect = sect.parent
    if sect.depth == depth:
        return sect
    raise SyntaxError()