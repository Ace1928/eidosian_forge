from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TestEnvironmentCatalog(_messages.Message):
    """A description of a test environment.

  Fields:
    androidDeviceCatalog: Supported Android devices.
    deviceIpBlockCatalog: The IP blocks used by devices in the test
      environment.
    iosDeviceCatalog: Supported iOS devices.
    networkConfigurationCatalog: Supported network configurations.
    softwareCatalog: The software test environment provided by
      TestExecutionService.
  """
    androidDeviceCatalog = _messages.MessageField('AndroidDeviceCatalog', 1)
    deviceIpBlockCatalog = _messages.MessageField('DeviceIpBlockCatalog', 2)
    iosDeviceCatalog = _messages.MessageField('IosDeviceCatalog', 3)
    networkConfigurationCatalog = _messages.MessageField('NetworkConfigurationCatalog', 4)
    softwareCatalog = _messages.MessageField('ProvidedSoftwareCatalog', 5)