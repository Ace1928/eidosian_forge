from pprint import pformat
from six import iteritems
import re
@verb.setter
def verb(self, verb):
    """
        Sets the verb of this V1beta1ResourceAttributes.
        Verb is a kubernetes resource API verb, like: get, list, watch, create,
        update, delete, proxy.  "*" means all.

        :param verb: The verb of this V1beta1ResourceAttributes.
        :type: str
        """
    self._verb = verb