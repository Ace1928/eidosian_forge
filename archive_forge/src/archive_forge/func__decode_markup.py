from collections import Counter
import os
import re
import sys
import traceback
import warnings
from .builder import (
from .dammit import UnicodeDammit
from .element import (
@classmethod
def _decode_markup(cls, markup):
    """Ensure `markup` is bytes so it's safe to send into warnings.warn.

        TODO: warnings.warn had this problem back in 2010 but it might not
        anymore.
        """
    if isinstance(markup, bytes):
        decoded = markup.decode('utf-8', 'replace')
    else:
        decoded = markup
    return decoded