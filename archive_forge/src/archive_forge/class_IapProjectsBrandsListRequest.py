from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IapProjectsBrandsListRequest(_messages.Message):
    """A IapProjectsBrandsListRequest object.

  Fields:
    parent: Required. GCP Project number/id. In the following format:
      projects/{project_number/id}.
  """
    parent = _messages.StringField(1, required=True)