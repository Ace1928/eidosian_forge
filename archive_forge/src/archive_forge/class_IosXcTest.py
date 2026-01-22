from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IosXcTest(_messages.Message):
    """A test of an iOS application that uses the XCTest framework.

  Fields:
    bundleId: Bundle ID of the app.
    xcodeVersion: Xcode version that the test was run with.
  """
    bundleId = _messages.StringField(1)
    xcodeVersion = _messages.StringField(2)