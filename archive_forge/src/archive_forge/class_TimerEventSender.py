import abc
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import api as vrrp_api
class TimerEventSender(Timer):

    def __init__(self, app, ev_cls):
        super(TimerEventSender, self).__init__(self._timeout)
        self._app = app
        self._ev_cls = ev_cls

    def _timeout(self):
        self._app.send_event(self._app.name, self._ev_cls())