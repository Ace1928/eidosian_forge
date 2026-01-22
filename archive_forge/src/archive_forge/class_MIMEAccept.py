from __future__ import annotations
import codecs
import re
from .structures import ImmutableList
class MIMEAccept(Accept):
    """Like :class:`Accept` but with special methods and behavior for
    mimetypes.
    """

    def _specificity(self, value):
        return tuple((x != '*' for x in _mime_split_re.split(value)))

    def _value_matches(self, value, item):
        if '/' not in item:
            return False
        if '/' not in value:
            raise ValueError(f'invalid mimetype {value!r}')
        normalized_value = _normalize_mime(value)
        value_type, value_subtype = normalized_value[:2]
        value_params = sorted(normalized_value[2:])
        if value_type == '*' and value_subtype != '*':
            raise ValueError(f'invalid mimetype {value!r}')
        normalized_item = _normalize_mime(item)
        item_type, item_subtype = normalized_item[:2]
        item_params = sorted(normalized_item[2:])
        if item_type == '*' and item_subtype != '*':
            return False
        return (item_type == '*' and item_subtype == '*' or (value_type == '*' and value_subtype == '*')) or (item_type == value_type and (item_subtype == '*' or value_subtype == '*' or (item_subtype == value_subtype and item_params == value_params)))

    @property
    def accept_html(self):
        """True if this object accepts HTML."""
        return 'text/html' in self or 'application/xhtml+xml' in self or self.accept_xhtml

    @property
    def accept_xhtml(self):
        """True if this object accepts XHTML."""
        return 'application/xhtml+xml' in self or 'application/xml' in self

    @property
    def accept_json(self):
        """True if this object accepts JSON."""
        return 'application/json' in self