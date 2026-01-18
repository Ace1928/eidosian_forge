from pprint import pformat
from six import iteritems
import re
@verbs.setter
def verbs(self, verbs):
    """
        Sets the verbs of this V1beta1ResourceRule.
        Verb is a list of kubernetes resource API verbs, like: get, list, watch,
        create, update, delete, proxy.  "*" means all.

        :param verbs: The verbs of this V1beta1ResourceRule.
        :type: list[str]
        """
    if verbs is None:
        raise ValueError('Invalid value for `verbs`, must not be `None`')
    self._verbs = verbs