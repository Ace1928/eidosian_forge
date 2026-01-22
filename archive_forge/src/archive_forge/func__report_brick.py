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
@staticmethod
def _report_brick(name, app):
    LOG.debug('BRICK %s', name)
    for ev_cls, list_ in app.observers.items():
        LOG.debug('  PROVIDES %s TO %s', ev_cls.__name__, list_)
    for ev_cls in app.event_handlers.keys():
        LOG.debug('  CONSUMES %s', ev_cls.__name__)