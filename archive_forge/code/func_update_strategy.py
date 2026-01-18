from pprint import pformat
from six import iteritems
import re
@update_strategy.setter
def update_strategy(self, update_strategy):
    """
        Sets the update_strategy of this V1beta2StatefulSetSpec.
        updateStrategy indicates the StatefulSetUpdateStrategy that will be
        employed to update Pods in the StatefulSet when a revision is made to
        Template.

        :param update_strategy: The update_strategy of this
        V1beta2StatefulSetSpec.
        :type: V1beta2StatefulSetUpdateStrategy
        """
    self._update_strategy = update_strategy