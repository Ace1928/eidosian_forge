import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def _get_required_string(parsed, flags):
    """Gets the required string and related info of a parsed pattern."""
    req_offset, required = parsed.get_required_string(bool(flags & REVERSE))
    if required:
        required.required = True
        if req_offset >= UNLIMITED:
            req_offset = -1
        req_flags = required.case_flags
        if not flags & UNICODE:
            req_flags &= ~UNICODE
        req_chars = required.folded_characters
    else:
        req_offset = 0
        req_chars = ()
        req_flags = 0
    return (req_offset, req_chars, req_flags)