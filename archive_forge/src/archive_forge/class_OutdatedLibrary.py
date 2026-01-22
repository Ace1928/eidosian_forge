from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OutdatedLibrary(_messages.Message):
    """Information reported for an outdated library.

  Fields:
    learnMoreUrls: URLs to learn more information about the vulnerabilities in
      the library.
    libraryName: The name of the outdated library.
    version: The version number.
  """
    learnMoreUrls = _messages.StringField(1, repeated=True)
    libraryName = _messages.StringField(2)
    version = _messages.StringField(3)