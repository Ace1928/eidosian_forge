from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsUrlListsPatchRequest(_messages.Message):
    """A NetworksecurityProjectsLocationsUrlListsPatchRequest object.

  Fields:
    name: Required. Name of the resource provided by the user. Name is of the
      form projects/{project}/locations/{location}/urlLists/{url_list}
      url_list should match the pattern:(^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$).
    updateMask: Optional. Field mask is used to specify the fields to be
      overwritten in the UrlList resource by the update. The fields specified
      in the update_mask are relative to the resource, not the full request. A
      field will be overwritten if it is in the mask. If the user does not
      provide a mask then all fields will be overwritten.
    urlList: A UrlList resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    updateMask = _messages.StringField(2)
    urlList = _messages.MessageField('UrlList', 3)