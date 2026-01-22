from __future__ import unicode_literals
import re
import pybtex.io
from pybtex.bibtex.interpreter import (
from pybtex.scanner import (
Strip the commented part of the line."

    >>> print(strip_comment('a normal line'))
    a normal line
    >>> print(strip_comment('%'))
    <BLANKLINE>
    >>> print(strip_comment('%comment'))
    <BLANKLINE>
    >>> print(strip_comment('trailing%'))
    trailing
    >>> print(strip_comment('a normal line% and a comment'))
    a normal line
    >>> print(strip_comment('"100% compatibility" is a myth'))
    "100% compatibility" is a myth
    >>> print(strip_comment('"100% compatibility" is a myth% or not?'))
    "100% compatibility" is a myth

    