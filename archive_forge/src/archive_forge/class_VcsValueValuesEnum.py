from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class VcsValueValuesEnum(_messages.Enum):
    """The version control system of the repo.

    Values:
      VCS_UNSPECIFIED: No version control system was specified.
      GIT: The Git version control system.
    """
    VCS_UNSPECIFIED = 0
    GIT = 1