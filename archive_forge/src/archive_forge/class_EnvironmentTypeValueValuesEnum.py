from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class EnvironmentTypeValueValuesEnum(_messages.Enum):
    """Required. The type of environment that should be listed.

    Values:
      ENVIRONMENT_TYPE_UNSPECIFIED: Do not use. For proto versioning only.
      ANDROID: A device running a version of the Android OS.
      IOS: A device running a version of iOS.
      NETWORK_CONFIGURATION: A network configuration to use when running a
        test.
      PROVIDED_SOFTWARE: The software environment provided by
        TestExecutionService.
      DEVICE_IP_BLOCKS: The IP blocks used by devices in the test environment.
    """
    ENVIRONMENT_TYPE_UNSPECIFIED = 0
    ANDROID = 1
    IOS = 2
    NETWORK_CONFIGURATION = 3
    PROVIDED_SOFTWARE = 4
    DEVICE_IP_BLOCKS = 5