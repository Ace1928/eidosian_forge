import os
import json
from parlai.core.opt import Opt
from parlai.core.teachers import DialogTeacher
from parlai.utils.misc import warn_once
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric, BleuMetric
import parlai.tasks.google_sgd.build as build_
def _delex(self, text, slots):
    delex = text
    for slot, values in slots.items():
        assert isinstance(values, list)
        for value in values:
            delex = delex.replace(value, slot)
    return delex