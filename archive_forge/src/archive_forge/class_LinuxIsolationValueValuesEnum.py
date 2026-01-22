from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LinuxIsolationValueValuesEnum(_messages.Enum):
    """linux_isolation allows overriding the docker runtime used for
    containers started on Linux.

    Values:
      LINUX_ISOLATION_UNSPECIFIED: Default value. Will be using Linux default
        runtime.
      GVISOR: Use gVisor runsc runtime.
      OFF: Use stardard Linux runtime. This has the same behaviour as
        unspecified, but it can be used to revert back from gVisor.
    """
    LINUX_ISOLATION_UNSPECIFIED = 0
    GVISOR = 1
    OFF = 2