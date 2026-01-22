from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LegacyRedirectionStateValueValuesEnum(_messages.Enum):
    """The redirection state of the legacy repositories in this project.

    Values:
      REDIRECTION_STATE_UNSPECIFIED: No redirection status has been set.
      REDIRECTION_FROM_GCR_IO_DISABLED: Redirection is disabled.
      REDIRECTION_FROM_GCR_IO_ENABLED: Redirection is enabled.
      REDIRECTION_FROM_GCR_IO_FINALIZED: Redirection is enabled, and has been
        finalized so cannot be reverted.
      REDIRECTION_FROM_GCR_IO_PARTIAL: Redirection is partially enabled.
      REDIRECTION_FROM_GCR_IO_ENABLED_AND_COPYING: Redirection is enabled and
        missing images are copied from GCR
      REDIRECTION_FROM_GCR_IO_PARTIAL_AND_COPYING: Redirection is partially
        enabled and missing images are copied from GCR
    """
    REDIRECTION_STATE_UNSPECIFIED = 0
    REDIRECTION_FROM_GCR_IO_DISABLED = 1
    REDIRECTION_FROM_GCR_IO_ENABLED = 2
    REDIRECTION_FROM_GCR_IO_FINALIZED = 3
    REDIRECTION_FROM_GCR_IO_PARTIAL = 4
    REDIRECTION_FROM_GCR_IO_ENABLED_AND_COPYING = 5
    REDIRECTION_FROM_GCR_IO_PARTIAL_AND_COPYING = 6