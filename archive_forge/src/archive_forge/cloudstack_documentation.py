from libcloud.utils.misc import reverse_dict
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider

        @inherits: :class:`Driver.create_balancer`

        :param location: Location
        :type  location: :class:`NodeLocation`

        :param private_port: Private port
        :type  private_port: ``int``

        :param network_id: The guest network this rule will be created for.
        :type  network_id: ``str``
        