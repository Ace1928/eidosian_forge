from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityDevicesDeviceUsersClientStatesGetRequest(_messages.Message):
    """A CloudidentityDevicesDeviceUsersClientStatesGetRequest object.

  Fields:
    customer: Optional. [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      customer. If you're using this API for your own organization, use
      `customers/my_customer` If you're using this API to manage another
      organization, use `customers/{customer}`, where customer is the customer
      to whom the device belongs.
    name: Required. [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      ClientState in format:
      `devices/{device}/deviceUsers/{device_user}/clientStates/{partner}`,
      where `device` is the unique ID assigned to the Device, `device_user` is
      the unique ID assigned to the User and `partner` identifies the partner
      storing the data. To get the client state for devices belonging to your
      own organization, the `partnerId` is in the format:
      `customerId-*anystring*`. Where the `customerId` is your organization's
      customer ID and `anystring` is any suffix. This suffix is used in
      setting up Custom Access Levels in Context-Aware Access. You may use
      `my_customer` instead of the customer ID for devices managed by your own
      organization. You may specify `-` in place of the `{device}`, so the
      ClientState resource name can be:
      `devices/-/deviceUsers/{device_user_resource}/clientStates/{partner}`.
  """
    customer = _messages.StringField(1)
    name = _messages.StringField(2, required=True)