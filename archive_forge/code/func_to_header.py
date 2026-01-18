from __future__ import annotations
import base64
import binascii
import typing as t
from ..http import dump_header
from ..http import parse_dict_header
from ..http import quote_header_value
from .structures import CallbackDict
def to_header(self) -> str:
    """Produce a ``WWW-Authenticate`` header value representing this data."""
    if self.token is not None:
        return f'{self.type.title()} {self.token}'
    if self.type == 'digest':
        items = []
        for key, value in self.parameters.items():
            if key in {'realm', 'domain', 'nonce', 'opaque', 'qop'}:
                value = quote_header_value(value, allow_token=False)
            else:
                value = quote_header_value(value)
            items.append(f'{key}={value}')
        return f'Digest {', '.join(items)}'
    return f'{self.type.title()} {dump_header(self.parameters)}'