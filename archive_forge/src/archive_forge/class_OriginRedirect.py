from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OriginRedirect(_messages.Message):
    """The options for following redirects from the origin.

  Enums:
    RedirectConditionsValueListEntryValuesEnum:

  Fields:
    redirectConditions: Optional. The set of HTTP redirect response codes that
      the CDN follows.
  """

    class RedirectConditionsValueListEntryValuesEnum(_messages.Enum):
        """RedirectConditionsValueListEntryValuesEnum enum type.

    Values:
      REDIRECT_CONDITIONS_UNSPECIFIED: It is an error to specify
        `REDIRECT_CONDITIONS_UNSPECIFIED`.
      MOVED_PERMANENTLY: Follow redirect on an `HTTP 301` error.
      FOUND: Follow redirect on an `HTTP 302` error.
      SEE_OTHER: Follow redirect on an `HTTP 303` error.
      TEMPORARY_REDIRECT: Follow redirect on an `HTTP 307` error.
      PERMANENT_REDIRECT: Follow redirect on an `HTTP 308` error.
    """
        REDIRECT_CONDITIONS_UNSPECIFIED = 0
        MOVED_PERMANENTLY = 1
        FOUND = 2
        SEE_OTHER = 3
        TEMPORARY_REDIRECT = 4
        PERMANENT_REDIRECT = 5
    redirectConditions = _messages.EnumField('RedirectConditionsValueListEntryValuesEnum', 1, repeated=True)