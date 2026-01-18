from pprint import pformat
from six import iteritems
import re
@topology_keys.setter
def topology_keys(self, topology_keys):
    """
        Sets the topology_keys of this V1beta1CSINodeDriver.
        topologyKeys is the list of keys supported by the driver. When a driver
        is initialized on a cluster, it provides a set of topology keys that it
        understands (e.g. "company.com/zone", "company.com/region"). When a
        driver is initialized on a node, it provides the same topology keys
        along with values. Kubelet will expose these topology keys as labels on
        its own node object. When Kubernetes does topology aware provisioning,
        it can use this list to determine which labels it should retrieve from
        the node object and pass back to the driver. It is possible for
        different nodes to use different topology keys. This can be empty if
        driver does not support topology.

        :param topology_keys: The topology_keys of this V1beta1CSINodeDriver.
        :type: list[str]
        """
    self._topology_keys = topology_keys