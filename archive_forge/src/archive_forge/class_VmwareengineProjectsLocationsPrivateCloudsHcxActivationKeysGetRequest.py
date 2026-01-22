from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsPrivateCloudsHcxActivationKeysGetRequest(_messages.Message):
    """A VmwareengineProjectsLocationsPrivateCloudsHcxActivationKeysGetRequest
  object.

  Fields:
    name: Required. The resource name of the HCX activation key to retrieve.
      Resource names are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/us-central1/privateClouds/my-
      cloud/hcxActivationKeys/my-key`
  """
    name = _messages.StringField(1, required=True)