from pprint import pformat
from six import iteritems
import re
@tolerations.setter
def tolerations(self, tolerations):
    """
        Sets the tolerations of this V1PodSpec.
        If specified, the pod's tolerations.

        :param tolerations: The tolerations of this V1PodSpec.
        :type: list[V1Toleration]
        """
    self._tolerations = tolerations