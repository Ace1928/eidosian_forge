from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityDevicesDeviceUsersClientStatesPatchRequest(_messages.Message):
    """A CloudidentityDevicesDeviceUsersClientStatesPatchRequest object.

  Fields:
    customer: Optional. [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      customer. If you're using this API for your own organization, use
      `customers/my_customer` If you're using this API to manage another
      organization, use `customers/{customer}`, where customer is the customer
      to whom the device belongs.
    googleAppsCloudidentityDevicesV1ClientState: A
      GoogleAppsCloudidentityDevicesV1ClientState resource to be passed as the
      request body.
    name: Output only. [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      ClientState in format:
      `devices/{device}/deviceUsers/{device_user}/clientState/{partner}`,
      where partner corresponds to the partner storing the data. For partners
      belonging to the "BeyondCorp Alliance", this is the partner ID specified
      to you by Google. For all other callers, this is a string of the form:
      `{customer}-suffix`, where `customer` is your customer ID. The *suffix*
      is any string the caller specifies. This string will be displayed
      verbatim in the administration console. This suffix is used in setting
      up Custom Access Levels in Context-Aware Access. Your organization's
      customer ID can be obtained from the URL: `GET
      https://www.googleapis.com/admin/directory/v1/customers/my_customer` The
      `id` field in the response contains the customer ID starting with the
      letter 'C'. The customer ID to be used in this API is the string after
      the letter 'C' (not including 'C')
    updateMask: Optional. Comma-separated list of fully qualified names of
      fields to be updated. If not specified, all updatable fields in
      ClientState are updated.
  """
    customer = _messages.StringField(1)
    googleAppsCloudidentityDevicesV1ClientState = _messages.MessageField('GoogleAppsCloudidentityDevicesV1ClientState', 2)
    name = _messages.StringField(3, required=True)
    updateMask = _messages.StringField(4)