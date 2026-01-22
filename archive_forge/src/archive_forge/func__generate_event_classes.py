import inspect
import logging
from os_ken import utils
from os_ken.controller import event
from os_ken.lib.packet import zebra
def _generate_event_classes():
    for zebra_cls in zebra.__dict__.values():
        if not inspect.isclass(zebra_cls) or not issubclass(zebra_cls, zebra._ZebraMessageBody) or zebra_cls.__name__.startswith('_'):
            continue
        ev = _define_event_class(zebra_cls)
        ZEBRA_EVENTS.append(ev)