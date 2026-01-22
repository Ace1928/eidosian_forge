from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileLocation(_messages.Message):
    """Indicates the location at which a package was found.

  Fields:
    filePath: For jars that are contained inside .war files, this filepath can
      indicate the path to war file combined with the path to jar file.
  """
    filePath = _messages.StringField(1)