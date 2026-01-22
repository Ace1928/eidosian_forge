from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthFailActionValueValuesEnum(_messages.Enum):
    """Action to take when users access resources that require
    authentication. Defaults to redirect.

    Values:
      AUTH_FAIL_ACTION_UNSPECIFIED: Not specified. AUTH_FAIL_ACTION_REDIRECT
        is assumed.
      AUTH_FAIL_ACTION_REDIRECT: Redirects user to "accounts.google.com". The
        user is redirected back to the application URL after signing in or
        creating an account.
      AUTH_FAIL_ACTION_UNAUTHORIZED: Rejects request with a 401 HTTP status
        code and an error message.
    """
    AUTH_FAIL_ACTION_UNSPECIFIED = 0
    AUTH_FAIL_ACTION_REDIRECT = 1
    AUTH_FAIL_ACTION_UNAUTHORIZED = 2