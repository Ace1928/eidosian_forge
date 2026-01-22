import time
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import monitor as vrrp_monitor
from os_ken.services.protocols.vrrp import router as vrrp_router
class VRRPInstance(object):

    def __init__(self, name, monitor_name, config, interface):
        super(VRRPInstance, self).__init__()
        self.name = name
        self.monitor_name = monitor_name
        self.config = config
        self.interface = interface
        self.state = None

    def state_changed(self, new_state):
        self.state = new_state