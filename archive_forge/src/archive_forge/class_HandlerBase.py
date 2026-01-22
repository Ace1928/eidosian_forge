from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import re
import string
import sys
import wsgiref.util
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import backendinfo
from googlecloudsdk.third_party.appengine._internal import six_subset
class HandlerBase(validation.Validated):
    """Base class for URLMap and ApiConfigHandler."""
    ATTRIBUTES = {URL: validation.Optional(_URL_REGEX), LOGIN: validation.Options(LOGIN_OPTIONAL, LOGIN_REQUIRED, LOGIN_ADMIN, default=LOGIN_OPTIONAL), AUTH_FAIL_ACTION: validation.Options(AUTH_FAIL_ACTION_REDIRECT, AUTH_FAIL_ACTION_UNAUTHORIZED, default=AUTH_FAIL_ACTION_REDIRECT), SECURE: validation.Options(SECURE_HTTP, SECURE_HTTPS, SECURE_HTTP_OR_HTTPS, SECURE_DEFAULT, default=SECURE_DEFAULT), HANDLER_SCRIPT: validation.Optional(_FILES_REGEX)}