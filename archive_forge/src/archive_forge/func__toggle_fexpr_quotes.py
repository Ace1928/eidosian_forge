import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import (
from mypy_extensions import trait
from black.comments import contains_pragma_comment
from black.lines import Line, append_leaves
from black.mode import Feature, Mode, Preview
from black.nodes import (
from black.rusty import Err, Ok, Result
from black.strings import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def _toggle_fexpr_quotes(fstring: str, old_quote: str) -> str:
    """
    Toggles quotes used in f-string expressions that are `old_quote`.

    f-string expressions can't contain backslashes, so we need to toggle the
    quotes if the f-string itself will end up using the same quote. We can
    simply toggle without escaping because, quotes can't be reused in f-string
    expressions. They will fail to parse.

    NOTE: If PEP 701 is accepted, above statement will no longer be true.
    Though if quotes can be reused, we can simply reuse them without updates or
    escaping, once Black figures out how to parse the new grammar.
    """
    new_quote = "'" if old_quote == '"' else '"'
    parts = []
    previous_index = 0
    for start, end in iter_fexpr_spans(fstring):
        parts.append(fstring[previous_index:start])
        parts.append(fstring[start:end].replace(old_quote, new_quote))
        previous_index = end
    parts.append(fstring[previous_index:])
    return ''.join(parts)