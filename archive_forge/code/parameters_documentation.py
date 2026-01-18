from __future__ import absolute_import, unicode_literals
import json
import os
import time
from oauthlib.common import add_params_to_qs, add_params_to_uri, unicode_type
from oauthlib.signals import scope_changed
from .errors import (InsecureTransportError, MismatchingStateError,
from .tokens import OAuth2Token
from .utils import is_secure_transport, list_to_scope, scope_to_list
Ensures token precence, token type, expiration and scope in params.