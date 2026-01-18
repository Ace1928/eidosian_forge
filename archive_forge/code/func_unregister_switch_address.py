import netaddr
from os_ken.base import app_manager
from os_ken.lib import hub
from os_ken.lib import ip
from . import ofp_event
def unregister_switch_address(addr):
    """
    Unregister the given switch address.

    Unregisters the given switch address to let
    os_ken.controller.controller.OpenFlowController stop trying to initiate
    connection to switch.

    :param addr: A tuple of (host, port) pair of switch.
    """
    ofp_handler = app_manager.lookup_service_brick(ofp_event.NAME)
    if ofp_handler is None or ofp_handler.controller is None:
        return
    ofp_handler.controller.stop_client_loop(addr)