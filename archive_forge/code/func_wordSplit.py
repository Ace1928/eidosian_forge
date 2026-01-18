from unicodedata import category
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.rl_config import _FUZZ
from reportlab.lib.utils import isUnicode
import re
def wordSplit(word, maxWidths, fontName, fontSize, encoding='utf8'):
    """Attempts to break a word which lacks spaces into two parts, the first of which
    fits in the remaining space.  It is allowed to add hyphens or whatever it wishes.

    This is intended as a wrapper for some language- and user-choice-specific splitting
    algorithms.  It should only be called after line breaking on spaces, which covers western
    languages and is highly optimised already.  It works on the 'last unsplit word'.

    Presumably with further study one could write a Unicode splitting algorithm for text
    fragments whick was much faster.

    Courier characters should be 6 points wide.
    >>> wordSplit('HelloWorld', 30, 'Courier', 10)
    [[0.0, 'Hello'], [0.0, 'World']]
    >>> wordSplit('HelloWorld', 31, 'Courier', 10)
    [[1.0, 'Hello'], [1.0, 'World']]
    """
    if not isUnicode(word):
        uword = word.decode(encoding)
    else:
        uword = word
    charWidths = getCharWidths(uword, fontName, fontSize)
    lines = dumbSplit(uword, charWidths, maxWidths)
    if not isUnicode(word):
        lines2 = []
        for extraSpace, text in lines:
            lines2.append([extraSpace, text.encode(encoding)])
        lines = lines2
    return lines