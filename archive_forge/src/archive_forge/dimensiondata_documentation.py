from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
from libcloud.common.dimensiondata import (

        Get the default iRules available for a network domain

        :param network_domain_id: The ID of of a ``DimensionDataNetworkDomain``
        :type  network_domain_id: ``str``

        :rtype: `list` of :class:`DimensionDataDefaultiRule`
        