from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider

        Functions takes a named parameter that can be one or none of the
        following to filter returned items

        :param params: A sequence of comma separated keyword arguments
        and a value
            * id=
            * network_domain_id=
            * datacenter_id=
            * name=
            * state=
            * ssl_domain_certificate_id=
            * ssl_domain_certificate_name=
            * ssl_certificate_chain_id=
            * ssl_certificate_chain_name=
            * create_time=
        :return: `list` of :class: `NttCisSslssloffloadprofile`
        