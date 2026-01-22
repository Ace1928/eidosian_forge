from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsPrivateCloudsIdentitySourcesCreateRequest(_messages.Message):
    """A VmwareengineProjectsLocationsPrivateCloudsIdentitySourcesCreateRequest
  object.

  Fields:
    identitySource: A IdentitySource resource to be passed as the request
      body.
    identitySourceId: Required. The user-provided identifier of the new
      `IdentitySource`. This identifier must be unique among identity sources
      within the parent and becomes the final token in the name URI. The
      identifier must meet the following requirements: * Only contains 1-63
      alphanumeric characters and hyphens * Begins with an alphabetical
      character * Ends with a non-hyphen character * Not formatted as a UUID *
      Complies with [RFC 1034](https://datatracker.ietf.org/doc/html/rfc1034)
      (section 3.5)
    parent: Required. The resource name of the private cloud to create a new
      identity source in. Resource names are schemeless URIs that follow the
      conventions in https://cloud.google.com/apis/design/resource_names. For
      example: `projects/my-project/locations/us-central1-a/privateClouds/my-
      cloud`
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
    validateOnly: Optional. True if you want the request to be validated and
      not executed; false otherwise.
  """
    identitySource = _messages.MessageField('IdentitySource', 1)
    identitySourceId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)
    validateOnly = _messages.BooleanField(5)