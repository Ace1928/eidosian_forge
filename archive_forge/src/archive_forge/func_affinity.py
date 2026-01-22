from pprint import pformat
from six import iteritems
import re
@affinity.setter
def affinity(self, affinity):
    """
        Sets the affinity of this V1PodSpec.
        If specified, the pod's scheduling constraints

        :param affinity: The affinity of this V1PodSpec.
        :type: V1Affinity
        """
    self._affinity = affinity