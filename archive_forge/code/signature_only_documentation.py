from __future__ import absolute_import, unicode_literals
import logging
from .. import errors
from .base import BaseEndpoint
Validate a signed OAuth request.

        :param uri: The full URI of the token request.
        :param http_method: A valid HTTP verb, i.e. GET, POST, PUT, HEAD, etc.
        :param body: The request body as a string.
        :param headers: The request headers as a dict.
        :returns: A tuple of 2 elements.
                  1. True if valid, False otherwise.
                  2. An oauthlib.common.Request object.
        