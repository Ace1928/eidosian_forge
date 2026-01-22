from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SchemaValidationValueValuesEnum(_messages.Enum):
    """Customize how deployment manager will validate the resource against
    schema errors.

    Values:
      UNKNOWN: <no description>
      IGNORE: Ignore schema failures.
      IGNORE_WITH_WARNINGS: Ignore schema failures but display them as
        warnings.
      FAIL: Fail the resource if the schema is not valid, this is the default
        behavior.
    """
    UNKNOWN = 0
    IGNORE = 1
    IGNORE_WITH_WARNINGS = 2
    FAIL = 3