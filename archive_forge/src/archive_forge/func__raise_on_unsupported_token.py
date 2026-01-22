from __future__ import absolute_import, unicode_literals
import functools
import logging
from ..errors import (FatalClientError, OAuth2Error, ServerError,
def _raise_on_unsupported_token(self, request):
    """Raise on unsupported tokens."""
    if request.token_type_hint and request.token_type_hint in self.valid_token_types and (request.token_type_hint not in self.supported_token_types):
        raise UnsupportedTokenTypeError(request=request)