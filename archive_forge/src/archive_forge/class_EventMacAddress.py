import collections
from os_ken.base import app_manager
import os_ken.exception as os_ken_exc
from os_ken.controller import event
from os_ken.exception import NetworkNotFound, NetworkAlreadyExist
from os_ken.exception import PortAlreadyExist, PortNotFound, PortUnknown
class EventMacAddress(event.EventBase):
    """
    An event class for end-point MAC address registration.

    This event is generated when a end-point MAC address is updated
    by the REST API.
    An instance has at least the following attributes.

    =========== ===============================================================
    Attribute   Description
    =========== ===============================================================
    network_id  Network ID
    dpid        OpenFlow Datapath ID of the switch to which the port belongs.
    port_no     OpenFlow port number of the port
    mac_address The old MAC address of the port if add_del is False.  Otherwise
                the new MAC address.
    add_del     False if this event is a result of a port removal.  Otherwise
                True.
    =========== ===============================================================
    """

    def __init__(self, dpid, port_no, network_id, mac_address, add_del):
        super(EventMacAddress, self).__init__()
        assert network_id is not None
        assert mac_address is not None
        self.dpid = dpid
        self.port_no = port_no
        self.network_id = network_id
        self.mac_address = mac_address
        self.add_del = add_del