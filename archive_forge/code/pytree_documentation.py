from typing import (
from blib2to3.pgen2.grammar import Grammar
import sys
from io import StringIO

        Initializer.

        The argument is either a pattern or None.  If it is None, this
        only matches an empty sequence (effectively '$' in regex
        lingo).  If it is not None, this matches whenever the argument
        pattern doesn't have any matches.
        