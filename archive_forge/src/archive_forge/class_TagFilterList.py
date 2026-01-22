from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TagFilterList(_messages.Message):
    """List of tags to filter.

  Fields:
    tags: Tags to filter. Tags must be DICOM Data Elements, File Meta
      Elements, or Directory Structuring Elements, as defined in the [Registry
      of DICOM Data Elements] (http://dicom.nema.org/medical/dicom/current/out
      put/html/part06.html#table_6-1). They can be provided by "Keyword" or
      "Tag". For example, "PatientID", "00100010".
  """
    tags = _messages.StringField(1, repeated=True)