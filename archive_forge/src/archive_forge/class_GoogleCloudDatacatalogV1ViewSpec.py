from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1ViewSpec(_messages.Message):
    """Table view specification.

  Fields:
    viewQuery: Output only. The query that defines the table view.
  """
    viewQuery = _messages.StringField(1)