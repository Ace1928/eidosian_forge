from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SymlinkValueValuesEnum(_messages.Enum):
    """Specifies how symlinks should be handled by the transfer. By default,
    symlinks are not preserved. Only applicable to transfers involving POSIX
    file systems, and ignored for other transfers.

    Values:
      SYMLINK_UNSPECIFIED: Symlink behavior is unspecified.
      SYMLINK_SKIP: Do not preserve symlinks during a transfer job.
      SYMLINK_PRESERVE: Preserve symlinks during a transfer job.
    """
    SYMLINK_UNSPECIFIED = 0
    SYMLINK_SKIP = 1
    SYMLINK_PRESERVE = 2