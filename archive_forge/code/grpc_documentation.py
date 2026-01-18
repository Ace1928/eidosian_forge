from __future__ import absolute_import
import logging
import os
import six
from google.auth import environment_vars
from google.auth import exceptions
from google.auth.transport import _mtls_helper
from google.oauth2 import service_account
Indicates if the created SSL channel credentials is mutual TLS.