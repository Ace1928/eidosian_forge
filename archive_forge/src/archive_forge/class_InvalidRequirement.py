import urllib.parse
from typing import Any, List, Optional, Set
from ._parser import parse_requirement as _parse_requirement
from ._tokenizer import ParserSyntaxError
from .markers import Marker, _normalize_extra_values
from .specifiers import SpecifierSet
class InvalidRequirement(ValueError):
    """
    An invalid requirement was found, users should refer to PEP 508.
    """