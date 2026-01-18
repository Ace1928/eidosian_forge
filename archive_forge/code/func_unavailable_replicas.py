from pprint import pformat
from six import iteritems
import re
@unavailable_replicas.setter
def unavailable_replicas(self, unavailable_replicas):
    """
        Sets the unavailable_replicas of this V1beta2DeploymentStatus.
        Total number of unavailable pods targeted by this deployment. This is
        the total number of pods that are still required for the deployment to
        have 100% available capacity. They may either be pods that are running
        but not yet available or pods that still have not been created.

        :param unavailable_replicas: The unavailable_replicas of this
        V1beta2DeploymentStatus.
        :type: int
        """
    self._unavailable_replicas = unavailable_replicas