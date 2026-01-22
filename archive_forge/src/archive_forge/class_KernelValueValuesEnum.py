from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KernelValueValuesEnum(_messages.Enum):
    """Optional. Kernel

    Values:
      KERNEL_UNSPECIFIED: The kernel is unknown.
      PYTHON: Python kernel.
      SCALA: Scala kernel.
    """
    KERNEL_UNSPECIFIED = 0
    PYTHON = 1
    SCALA = 2