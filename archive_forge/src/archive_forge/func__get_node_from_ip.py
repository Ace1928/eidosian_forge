from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.compute.drivers.gce import GCEConnection, GCENodeDriver
def _get_node_from_ip(self, ip):
    """
        Return the node object that matches a given public IP address.

        :param  ip: Public IP address to search for
        :type   ip: ``str``

        :return:  Node object that has the given IP, or None if not found.
        :rtype:   :class:`Node` or None
        """
    all_nodes = self.gce.list_nodes(ex_zone='all')
    for node in all_nodes:
        if ip in node.public_ips:
            return node
    return None