from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IosTest(_messages.Message):
    """A iOS mobile test specification

  Fields:
    iosAppInfo: Information about the application under test.
    iosRoboTest: An iOS Robo test.
    iosTestLoop: An iOS test loop.
    iosXcTest: An iOS XCTest.
    testTimeout: Max time a test is allowed to run before it is automatically
      cancelled.
  """
    iosAppInfo = _messages.MessageField('IosAppInfo', 1)
    iosRoboTest = _messages.MessageField('IosRoboTest', 2)
    iosTestLoop = _messages.MessageField('IosTestLoop', 3)
    iosXcTest = _messages.MessageField('IosXcTest', 4)
    testTimeout = _messages.MessageField('Duration', 5)