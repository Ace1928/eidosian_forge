from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VideointelligenceProjectsLocationsOperationsDeleteRequest(_messages.Message):
    """A VideointelligenceProjectsLocationsOperationsDeleteRequest object.

  Fields:
    name: The name of the operation resource to be deleted.
  """
    name = _messages.StringField(1, required=True)