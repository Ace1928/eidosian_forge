from __future__ import absolute_import
from . import GrocLexer
from . import GrocParser
import antlr3
Raise an exception if the input fails to parse correctly.

    Overriding the default, which normally just prints a message to
    stderr.

    Arguments:
      msg: the error message

    Raises:
      GrocException: always.
    