from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GPUDriverConfig(_messages.Message):
    """A GPU driver configuration

  Fields:
    customGpuDriverPath: Optional. Specify a custom Cloud Storage path where
      the GPU driver is stored. If not specified, we'll automatically choose
      from official GPU drivers.
    enableGpuDriver: Optional. Whether the end user authorizes Google Cloud to
      install GPU driver on this VM instance. If this field is empty or set to
      false, the GPU driver won't be installed. Only applicable to instances
      with GPUs.
  """
    customGpuDriverPath = _messages.StringField(1)
    enableGpuDriver = _messages.BooleanField(2)