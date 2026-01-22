from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VpcaccessProjectsLocationsConnectorsDeleteRequest(_messages.Message):
    """A VpcaccessProjectsLocationsConnectorsDeleteRequest object.

  Fields:
    name: Required. Name of a Serverless VPC Access connector to delete.
  """
    name = _messages.StringField(1, required=True)