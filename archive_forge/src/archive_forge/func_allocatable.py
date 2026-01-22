from pprint import pformat
from six import iteritems
import re
@allocatable.setter
def allocatable(self, allocatable):
    """
        Sets the allocatable of this V1NodeStatus.
        Allocatable represents the resources of a node that are available for
        scheduling. Defaults to Capacity.

        :param allocatable: The allocatable of this V1NodeStatus.
        :type: dict(str, str)
        """
    self._allocatable = allocatable