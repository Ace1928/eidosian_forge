import boto
from datetime import datetime
from boto.resultset import ResultSet
from boto.ec2.ec2object import TaggedEC2Object

    Represents a VPN Connection

    :ivar id: The ID of the VPN connection.
    :ivar state: The current state of the VPN connection.
        Valid values: pending | available | deleting | deleted
    :ivar customer_gateway_configuration: The configuration information for the
        VPN connection's customer gateway (in the native XML format). This
        element is always present in the
        :class:`boto.vpc.VPCConnection.create_vpn_connection` response;
        however, it's present in the
        :class:`boto.vpc.VPCConnection.get_all_vpn_connections` response only
        if the VPN connection is in the pending or available state.
    :ivar type: The type of VPN connection (ipsec.1).
    :ivar customer_gateway_id: The ID of the customer gateway at your end of
        the VPN connection.
    :ivar vpn_gateway_id: The ID of the virtual private gateway
        at the AWS side of the VPN connection.
    :ivar tunnels: A list of the vpn tunnels (always 2)
    :ivar options: The option set describing the VPN connection.
    :ivar static_routes: A list of static routes associated with a VPN
        connection.

    