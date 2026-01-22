from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentaiProjectsLocationsFetchProcessorTypesRequest(_messages.Message):
    """A DocumentaiProjectsLocationsFetchProcessorTypesRequest object.

  Fields:
    parent: Required. The location of processor types to list. Format:
      `projects/{project}/locations/{location}`.
  """
    parent = _messages.StringField(1, required=True)