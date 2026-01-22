from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AddSubnetworkRequest(_messages.Message):
    """Request to create a subnetwork in a previously peered service network.

  Fields:
    consumer: Resource representing service consumer. It may be different from
      the project number in consumer network parameter in case of that network
      being a shared VPC network. In that case, Service Networking will
      validate that this resource belongs to that shared VPC.  Required. For
      example 'projects/123456'.
    consumerNetwork: Network name in the consumer project.   This network must
      have been already peered with a shared VPC network using
      CreateConnection method. Must be in a form
      'projects/{project}/global/networks/{network}'. {project} is a project
      number, as in '12345' {network} is network name.
    description: Description of the subnetwork.
    ipPrefixLength: The prefix length of the IP range. Use usual CIDR range
      notation. For example, '30' to provision subnet with x.x.x.x/30 CIDR
      range. Actual range will determined using reserved range for the
      consumer peered network and returned in the result.
    region: Cloud [region](/compute/docs/reference/rest/v1/regions) for the
      new subnetwork.
    subnetwork: Name for the new subnetwork. Must be a legal
      [subnetwork](compute/docs/reference/rest/v1/subnetworks) name.
    subnetworkUsers: List of members that will be granted
      'compute.networkUser' role on the newly added subnetwork.
  """
    consumer = _messages.StringField(1)
    consumerNetwork = _messages.StringField(2)
    description = _messages.StringField(3)
    ipPrefixLength = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    region = _messages.StringField(5)
    subnetwork = _messages.StringField(6)
    subnetworkUsers = _messages.StringField(7, repeated=True)