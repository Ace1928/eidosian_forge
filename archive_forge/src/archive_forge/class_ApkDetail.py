from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ApkDetail(_messages.Message):
    """Android application details based on application manifest and archive
  contents.

  Fields:
    apkManifest: A ApkManifest attribute.
  """
    apkManifest = _messages.MessageField('ApkManifest', 1)