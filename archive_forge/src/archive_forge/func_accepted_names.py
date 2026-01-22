from pprint import pformat
from six import iteritems
import re
@accepted_names.setter
def accepted_names(self, accepted_names):
    """
        Sets the accepted_names of this V1beta1CustomResourceDefinitionStatus.
        AcceptedNames are the names that are actually being used to serve
        discovery They may be different than the names in spec.

        :param accepted_names: The accepted_names of this
        V1beta1CustomResourceDefinitionStatus.
        :type: V1beta1CustomResourceDefinitionNames
        """
    if accepted_names is None:
        raise ValueError('Invalid value for `accepted_names`, must not be `None`')
    self._accepted_names = accepted_names