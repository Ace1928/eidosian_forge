from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1EntryOverview(_messages.Message):
    """Entry overview fields for rich text descriptions of entries.

  Fields:
    overview: Entry overview with support for rich text. The overview must
      only contain Unicode characters, and should be formatted using HTML. The
      maximum length is 10 MiB as this value holds HTML descriptions including
      encoded images. The maximum length of the text without images is 100
      KiB.
  """
    overview = _messages.StringField(1)