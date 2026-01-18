from pprint import pformat
from six import iteritems
import re
@volumes_attached.setter
def volumes_attached(self, volumes_attached):
    """
        Sets the volumes_attached of this V1NodeStatus.
        List of volumes that are attached to the node.

        :param volumes_attached: The volumes_attached of this V1NodeStatus.
        :type: list[V1AttachedVolume]
        """
    self._volumes_attached = volumes_attached