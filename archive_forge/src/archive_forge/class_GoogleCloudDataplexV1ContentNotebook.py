from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1ContentNotebook(_messages.Message):
    """Configuration for Notebook content.

  Enums:
    KernelTypeValueValuesEnum: Required. Kernel Type of the notebook.

  Fields:
    kernelType: Required. Kernel Type of the notebook.
  """

    class KernelTypeValueValuesEnum(_messages.Enum):
        """Required. Kernel Type of the notebook.

    Values:
      KERNEL_TYPE_UNSPECIFIED: Kernel Type unspecified.
      PYTHON3: Python 3 Kernel.
    """
        KERNEL_TYPE_UNSPECIFIED = 0
        PYTHON3 = 1
    kernelType = _messages.EnumField('KernelTypeValueValuesEnum', 1)