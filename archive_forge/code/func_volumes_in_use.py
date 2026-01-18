from pprint import pformat
from six import iteritems
import re
@volumes_in_use.setter
def volumes_in_use(self, volumes_in_use):
    """
        Sets the volumes_in_use of this V1NodeStatus.
        List of attachable volumes in use (mounted) by the node.

        :param volumes_in_use: The volumes_in_use of this V1NodeStatus.
        :type: list[str]
        """
    self._volumes_in_use = volumes_in_use