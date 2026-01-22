from collections import Counter
from threading import Timer
import logging
import inspect
from ..core import MachineError, listify, State
class CustomState(type('CustomState', args, {}), cls.state_cls):
    """ The decorated State. It is based on the State class used by the decorated Machine. """