import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.directconnect import exceptions
from boto.compat import json
def allocate_connection_on_interconnect(self, bandwidth, connection_name, owner_account, interconnect_id, vlan):
    """
        Creates a hosted connection on an interconnect.

        Allocates a VLAN number and a specified amount of bandwidth
        for use by a hosted connection on the given interconnect.

        :type bandwidth: string
        :param bandwidth: Bandwidth of the connection.
        Example: " 500Mbps "

        Default: None

        :type connection_name: string
        :param connection_name: Name of the provisioned connection.
        Example: " 500M Connection to AWS "

        Default: None

        :type owner_account: string
        :param owner_account: Numeric account Id of the customer for whom the
            connection will be provisioned.
        Example: 123443215678

        Default: None

        :type interconnect_id: string
        :param interconnect_id: ID of the interconnect on which the connection
            will be provisioned.
        Example: dxcon-456abc78

        Default: None

        :type vlan: integer
        :param vlan: The dedicated VLAN provisioned to the connection.
        Example: 101

        Default: None

        """
    params = {'bandwidth': bandwidth, 'connectionName': connection_name, 'ownerAccount': owner_account, 'interconnectId': interconnect_id, 'vlan': vlan}
    return self.make_request(action='AllocateConnectionOnInterconnect', body=json.dumps(params))