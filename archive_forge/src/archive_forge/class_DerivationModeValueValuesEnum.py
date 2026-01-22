from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DerivationModeValueValuesEnum(_messages.Enum):
    """DEPRECATED

    Values:
      SOURCE_DERIVATION_MODE_UNKNOWN: The source derivation is unknown, or
        unspecified.
      SOURCE_DERIVATION_MODE_INDEPENDENT: Produce a completely independent
        Source with no base.
      SOURCE_DERIVATION_MODE_CHILD_OF_CURRENT: Produce a Source based on the
        Source being split.
      SOURCE_DERIVATION_MODE_SIBLING_OF_CURRENT: Produce a Source based on the
        base of the Source being split.
    """
    SOURCE_DERIVATION_MODE_UNKNOWN = 0
    SOURCE_DERIVATION_MODE_INDEPENDENT = 1
    SOURCE_DERIVATION_MODE_CHILD_OF_CURRENT = 2
    SOURCE_DERIVATION_MODE_SIBLING_OF_CURRENT = 3