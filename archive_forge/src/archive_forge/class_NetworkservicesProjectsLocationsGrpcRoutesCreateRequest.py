from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsGrpcRoutesCreateRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsGrpcRoutesCreateRequest object.

  Fields:
    grpcRoute: A GrpcRoute resource to be passed as the request body.
    grpcRouteId: Required. Short name of the GrpcRoute resource to be created.
    parent: Required. The parent resource of the GrpcRoute. Must be in the
      format `projects/*/locations/global`.
  """
    grpcRoute = _messages.MessageField('GrpcRoute', 1)
    grpcRouteId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)