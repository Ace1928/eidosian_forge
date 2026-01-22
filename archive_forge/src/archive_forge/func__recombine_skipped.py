from __future__ import unicode_literals
import datetime
import re
import string
import time
import warnings
from calendar import monthrange
from io import StringIO
import six
from six import integer_types, text_type
from decimal import Decimal
from warnings import warn
from .. import relativedelta
from .. import tz
def _recombine_skipped(self, tokens, skipped_idxs):
    """
        >>> tokens = ["foo", " ", "bar", " ", "19June2000", "baz"]
        >>> skipped_idxs = [0, 1, 2, 5]
        >>> _recombine_skipped(tokens, skipped_idxs)
        ["foo bar", "baz"]
        """
    skipped_tokens = []
    for i, idx in enumerate(sorted(skipped_idxs)):
        if i > 0 and idx - 1 == skipped_idxs[i - 1]:
            skipped_tokens[-1] = skipped_tokens[-1] + tokens[idx]
        else:
            skipped_tokens.append(tokens[idx])
    return skipped_tokens