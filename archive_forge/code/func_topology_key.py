from pprint import pformat
from six import iteritems
import re
@topology_key.setter
def topology_key(self, topology_key):
    """
        Sets the topology_key of this V1PodAffinityTerm.
        This pod should be co-located (affinity) or not co-located
        (anti-affinity) with the pods matching the labelSelector in the
        specified namespaces, where co-located is defined as running on a node
        whose value of the label with key topologyKey matches that of any node
        on which any of the selected pods is running. Empty topologyKey is not
        allowed.

        :param topology_key: The topology_key of this V1PodAffinityTerm.
        :type: str
        """
    if topology_key is None:
        raise ValueError('Invalid value for `topology_key`, must not be `None`')
    self._topology_key = topology_key