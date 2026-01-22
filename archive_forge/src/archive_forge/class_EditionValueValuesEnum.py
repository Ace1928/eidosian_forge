from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EditionValueValuesEnum(_messages.Enum):
    """Optional. The edition of the given Cloud SQL instance.

    Values:
      EDITION_UNSPECIFIED: The instance did not specify the edition.
      ENTERPRISE: The instance is an enterprise edition.
      ENTERPRISE_PLUS: The instance is an enterprise plus edition.
    """
    EDITION_UNSPECIFIED = 0
    ENTERPRISE = 1
    ENTERPRISE_PLUS = 2