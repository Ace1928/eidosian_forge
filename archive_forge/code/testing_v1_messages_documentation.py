from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
Required. The type of environment that should be listed.

    Values:
      ENVIRONMENT_TYPE_UNSPECIFIED: Do not use. For proto versioning only.
      ANDROID: A device running a version of the Android OS.
      IOS: A device running a version of iOS.
      NETWORK_CONFIGURATION: A network configuration to use when running a
        test.
      PROVIDED_SOFTWARE: The software environment provided by
        TestExecutionService.
      DEVICE_IP_BLOCKS: The IP blocks used by devices in the test environment.
    