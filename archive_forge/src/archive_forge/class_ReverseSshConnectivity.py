from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReverseSshConnectivity(_messages.Message):
    """The details needed to configure a reverse SSH tunnel between the source
  and destination databases. These details will be used when calling the
  generateSshScript method (see https://cloud.google.com/database-migration/do
  cs/reference/rest/v1alpha2/projects.locations.migrationJobs/generateSshScrip
  t) to produce the script that will help set up the reverse SSH tunnel, and
  to set up the VPC peering between the Cloud SQL private network and the VPC.

  Fields:
    vm: The name of the virtual machine (Compute Engine) used as the bastion
      server for the SSH tunnel.
    vmIp: Required. The IP of the virtual machine (Compute Engine) used as the
      bastion server for the SSH tunnel.
    vmPort: Required. The forwarding port of the virtual machine (Compute
      Engine) used as the bastion server for the SSH tunnel.
    vpc: The name of the VPC to peer with the Cloud SQL private network.
  """
    vm = _messages.StringField(1)
    vmIp = _messages.StringField(2)
    vmPort = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    vpc = _messages.StringField(4)