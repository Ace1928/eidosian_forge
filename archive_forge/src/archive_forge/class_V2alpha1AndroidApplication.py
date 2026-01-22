from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V2alpha1AndroidApplication(_messages.Message):
    """Identifier of an Android application for API key use.

  Fields:
    packageName: The package name of the application.
    sha1Fingerprint: The SHA1 fingerprint of the application. For example,
      both sha1 formats are acceptable as input:
      DA:39:A3:EE:5E:6B:4B:0D:32:55:BF:EF:95:60:18:90:AF:D8:07:09 or
      DA39A3EE5E6B4B0D3255BFEF95601890AFD80709. Output format is the latter.
  """
    packageName = _messages.StringField(1)
    sha1Fingerprint = _messages.StringField(2)