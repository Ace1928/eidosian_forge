import netaddr
from os_ken.base import app_manager
from os_ken.lib import hub
from os_ken.lib import ip
from . import ofp_event
def _retry_loop():
    while True:
        if ofp_handler is not None and ofp_handler.controller is not None:
            for a, i in _TMP_ADDRESSES.items():
                ofp_handler.controller.spawn_client_loop(a, i)
                hub.sleep(1)
            break
        hub.sleep(1)