from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentaiProjectsLocationsProcessorTypesGetRequest(_messages.Message):
    """A DocumentaiProjectsLocationsProcessorTypesGetRequest object.

  Fields:
    name: Required. The processor type resource name.
  """
    name = _messages.StringField(1, required=True)