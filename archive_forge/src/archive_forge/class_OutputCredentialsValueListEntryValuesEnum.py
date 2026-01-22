from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OutputCredentialsValueListEntryValuesEnum(_messages.Enum):
    """OutputCredentialsValueListEntryValuesEnum enum type.

    Values:
      OUTPUT_CREDENTIALS_UNSPECIFIED: An output credential is required.
      HEADER: Propagate attributes in the headers with "x-goog-iap-attr-"
        prefix.
      JWT: Propagate attributes in the JWT of the form: `"additional_claims":
        { "my_attribute": ["value1", "value2"] }`
      RCTOKEN: Propagate attributes in the RCToken of the form:
        `"additional_claims": { "my_attribute": ["value1", "value2"] }`
    """
    OUTPUT_CREDENTIALS_UNSPECIFIED = 0
    HEADER = 1
    JWT = 2
    RCTOKEN = 3