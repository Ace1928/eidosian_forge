from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsPrivateCloudsIdentitySourcesPatchRequest(_messages.Message):
    """A VmwareengineProjectsLocationsPrivateCloudsIdentitySourcesPatchRequest
  object.

  Fields:
    identitySource: A IdentitySource resource to be passed as the request
      body.
    name: Output only. Identifier. The resource name of this identity source.
      Resource names are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/us-central1-a/privateClouds/my-
      cloud/identitySources/my-identity-source`
    requestId: Optional. An identifier to identify requests. Specify a unique
      request ID so that if you must retry your request, the server will know
      to not execute the request again if it has already been executed. The
      server guarantees that a request doesn't result in creation of duplicate
      commitments for at least 60 minutes. Additionally, if there's a
      duplicate request ID, the response from the previous request will be
      returned. For example, consider a situation where you make an initial
      request and the request times out. If you make the request again with
      the same request ID, the server can check if the original operation with
      the same request ID was received, and if so, will ignore the second
      request. This prevents clients from accidentally creating duplicate
      commitments. The request ID must be a valid UUID with the exception that
      zero UUID is not supported (00000000-0000-0000-0000-000000000000).
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten in the `UpdateIdentitySource` resource by the update. The
      fields specified in the `updateMask` are relative to the resource, not
      the full request. A field will be overwritten if it is in the mask. If
      the user does not provide a mask then all fields will be overwritten.
    validateOnly: Optional. True if you want the request to be validated and
      not executed; false otherwise.
  """
    identitySource = _messages.MessageField('IdentitySource', 1)
    name = _messages.StringField(2, required=True)
    requestId = _messages.StringField(3)
    updateMask = _messages.StringField(4)
    validateOnly = _messages.BooleanField(5)