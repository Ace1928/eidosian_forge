from pprint import pformat
from six import iteritems
import re
@updated_replicas.setter
def updated_replicas(self, updated_replicas):
    """
        Sets the updated_replicas of this V1beta1StatefulSetStatus.
        updatedReplicas is the number of Pods created by the StatefulSet
        controller from the StatefulSet version indicated by updateRevision.

        :param updated_replicas: The updated_replicas of this
        V1beta1StatefulSetStatus.
        :type: int
        """
    self._updated_replicas = updated_replicas