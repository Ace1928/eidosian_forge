import time
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import monitor as vrrp_monitor
from os_ken.services.protocols.vrrp import router as vrrp_router
def _shutdown_loop(self):
    app_mgr = app_manager.AppManager.get_instance()
    while self.is_active or not self.shutdown.empty():
        instance = self.shutdown.get()
        app_mgr.uninstantiate(instance.name)
        app_mgr.uninstantiate(instance.monitor_name)
        del self._instances[instance.name]