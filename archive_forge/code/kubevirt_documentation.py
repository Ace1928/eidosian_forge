import json
import time
import hashlib
from datetime import datetime
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.kubernetes import (

        Each node has a single service of one type on which the exposed ports
        are described. If a service exists then the port declared will be
        exposed alongside the existing ones, set override_existing_ports=True
        to delete existing exposed ports and expose just the ones in the port
        variable.

        param node: the libcloud node for which the ports will be exposed
        type  node: libcloud `Node` class

        param ports: a list of dictionaries with keys --> values:
                     'port' --> port to be exposed on the service
                     'target_port' --> port on the pod/node, optional
                                       if empty then it gets the same
                                       value as 'port' value
                     'protocol' ---> either 'UDP' or 'TCP', defaults to TCP
                     'name' --> A name for the service
                     If ports is an empty `list` and a service exists of this
                     type then the service will be deleted.
        type  ports: `list` of `dict` where each `dict` has keys --> values:
                     'port' --> `int`
                     'target_port' --> `int`
                     'protocol' --> `str`
                     'name' --> `str`

        param service_type: Valid types are ClusterIP, NodePort, LoadBalancer
        type  service_type: `str`

        param cluster_ip: This can be set with an IP string value if you want
                          manually set the service's internal IP. If the value
                          is not correct the method will fail, this value can't
                          be updated.
        type  cluster_ip: `str`

        param override_existing_ports: Set to True if you want to delete the
                                       existing ports exposed by the service
                                       and keep just the ones declared in the
                                       present ports argument.
                                       By default it is false and if the
                                       service already exists the ports will be
                                       added to the existing ones.
        type  override_existing_ports: `boolean`
        