from __future__ import annotations
import unicodedata
import warnings
from typing import (
from ftfy import bad_codecs
from ftfy import chardata, fixes
from ftfy.badness import is_bad
from ftfy.formatting import display_ljust
class ExplanationStep(NamedTuple):
    """
    A step in an ExplainedText, explaining how to decode text.

    The possible actions are:

    - "encode": take in a string and encode it as bytes, with the given encoding
    - "decode": take in bytes and decode them as a string, with the given encoding
    - "transcode": convert bytes to bytes with a particular named function
    - "apply": convert str to str with a particular named function

    The `parameter` is the name of the encoding or function to use. If it's a
    function, it must appear in the FIXERS dictionary.
    """
    action: str
    parameter: str

    def __repr__(self) -> str:
        """
        Get the string representation of an ExplanationStep. We output the
        representation of the equivalent tuple, for simplicity.
        """
        return repr(tuple(self))