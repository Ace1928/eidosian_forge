from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OutOfBandDeploymentConfig(_messages.Message):
    """Config used to deploy an out of band appliance topology.

  Fields:
    applianceId: The id returned by our get appliance methods that identifies
      the appliance that will be deployed.
    applianceLoginSshKey: The public ssh key that will be used to log into the
      appliance.
    bootstrapParams: The bootstrap parameters that will be passed to appliance
      on first boot. The format for bootstrap parameters will depend on the
      appliance vendor and will vary. TODO (http://b/242734886): Add links to
      docs once they are in place.
    machineType: The GCP machine type the appliance vm will be installed on.
      Example: e2-standard-4 See
      https://cloud.google.com/compute/docs/machine-types for more
      information. TODO (http://b/242734886): Add links to docs once they are
      in place (in this case, verify the above link is what we want for the
      final link).
    managementSubnet: The name of an existing subnet the appliance's
      management interface should connect to. This must be a subnet that
      exists in the vpc specified in the 'management_vpc' field. This field
      must be set if 'management_vpc' is set.
    managementSubnetCidrRange: The cidr range the appliance's management
      interface should get its ip from. Either this should be set, or
      management_vpc and management_subnet should both be set. Example:
      25.72.12.0/24
    managementVpc: The name of an existing vpc that the appliance's management
      interface should connect to.
    maxInstances: The maximum number of instances of the appliance that should
      be running at any time.
    minInstances: The minimum number of instances of the appliance that should
      be running at any time.
    namingPrefix: A user specified string that will be prefixed to names of
      resources we create on their behalf.
    trafficSubnetCidrRange: The cidr range that specifies the traffic that
      should be directed to the appliance. Example: 25.72.12.0/24
    zones: The zones the appliance group will be deployed to. TODO
      (http://b/242734886): Add links to docs once they are in place.
  """
    applianceId = _messages.StringField(1)
    applianceLoginSshKey = _messages.StringField(2)
    bootstrapParams = _messages.StringField(3)
    machineType = _messages.StringField(4)
    managementSubnet = _messages.StringField(5)
    managementSubnetCidrRange = _messages.StringField(6)
    managementVpc = _messages.StringField(7)
    maxInstances = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    minInstances = _messages.IntegerField(9, variant=_messages.Variant.INT32)
    namingPrefix = _messages.StringField(10)
    trafficSubnetCidrRange = _messages.StringField(11)
    zones = _messages.StringField(12, repeated=True)