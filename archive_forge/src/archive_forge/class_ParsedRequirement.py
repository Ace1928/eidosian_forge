import ast
from typing import Any, List, NamedTuple, Optional, Tuple, Union
from ._tokenizer import DEFAULT_RULES, Tokenizer
class ParsedRequirement(NamedTuple):
    name: str
    url: str
    extras: List[str]
    specifier: str
    marker: Optional[MarkerList]