from __future__ import annotations
import unicodedata
import warnings
from typing import (
from ftfy import bad_codecs
from ftfy import chardata, fixes
from ftfy.badness import is_bad
from ftfy.formatting import display_ljust
@no_type_check
def apply_plan(text: str, plan: List[Tuple[str, str]]):
    """
    Apply a plan for fixing the encoding of text.

    The plan is a list of tuples of the form (operation, arg).

    `operation` is one of:

    - `'encode'`: convert a string to bytes, using `arg` as the encoding
    - `'decode'`: convert bytes to a string, using `arg` as the encoding
    - `'transcode'`: convert bytes to bytes, using the function named `arg`
    - `'apply'`: convert a string to a string, using the function named `arg`

    The functions that can be applied by 'transcode' and 'apply' are
    specifically those that appear in the dictionary named `FIXERS`. They
    can also can be imported from the `ftfy.fixes` module.

    Example::

        >>> mojibake = "schÃ¶n"
        >>> text, plan = fix_and_explain(mojibake)
        >>> apply_plan(mojibake, plan)
        'schön'
    """
    obj = text
    for operation, encoding in plan:
        if operation == 'encode':
            obj = obj.encode(encoding)
        elif operation == 'decode':
            obj = obj.decode(encoding)
        elif operation in ('transcode', 'apply'):
            if encoding in FIXERS:
                obj = FIXERS[encoding](obj)
            else:
                raise ValueError('Unknown function to apply: %s' % encoding)
        else:
            raise ValueError('Unknown plan step: %s' % operation)
    return obj