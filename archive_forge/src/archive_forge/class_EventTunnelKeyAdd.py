import collections
import logging
import os_ken.exception as os_ken_exc
from os_ken.base import app_manager
from os_ken.controller import event
class EventTunnelKeyAdd(EventTunnelKeyBase):
    """
    An event class for tunnel key registration.

    This event is generated when a tunnel key is registered or updated
    by the REST API.
    An instance has at least the following attributes.

    =========== ===============================================================
    Attribute   Description
    =========== ===============================================================
    network_id  Network ID
    tunnel_key  Tunnel Key
    =========== ===============================================================
    """

    def __init__(self, network_id, tunnel_key):
        super(EventTunnelKeyAdd, self).__init__(network_id, tunnel_key)