import inspect
import itertools
import logging
import sys
import os
import gc
from os_ken import cfg
from os_ken import utils
from os_ken.controller.handler import register_instance, get_dependent_services
from os_ken.controller.controller import Datapath
from os_ken.controller import event
from os_ken.controller.event import EventRequestBase, EventReplyBase
from os_ken.lib import hub
from os_ken.ofproto import ofproto_protocol
def uninstantiate(self, name):
    app = self.applications.pop(name)
    unregister_app(app)
    for app_ in SERVICE_BRICKS.values():
        app_.unregister_observer_all_event(name)
    app.stop()
    self._close(app)
    events = app.events
    if not events.empty():
        app.logger.debug('%s events remains %d', app.name, events.qsize())