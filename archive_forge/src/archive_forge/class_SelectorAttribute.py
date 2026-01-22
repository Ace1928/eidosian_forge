from __future__ import annotations
import copyreg
from .pretty import pretty
from typing import Any, Iterator, Hashable, Pattern, Iterable, Mapping
class SelectorAttribute(Immutable):
    """Selector attribute rule."""
    __slots__ = ('attribute', 'prefix', 'pattern', 'xml_type_pattern', '_hash')
    attribute: str
    prefix: str
    pattern: Pattern[str] | None
    xml_type_pattern: Pattern[str] | None

    def __init__(self, attribute: str, prefix: str, pattern: Pattern[str] | None, xml_type_pattern: Pattern[str] | None) -> None:
        """Initialize."""
        super().__init__(attribute=attribute, prefix=prefix, pattern=pattern, xml_type_pattern=xml_type_pattern)