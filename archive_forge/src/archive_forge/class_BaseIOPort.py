import random
import threading
import time
from .messages import Message
from .parser import Parser
class BaseIOPort(BaseInput, BaseOutput):

    def __init__(self, name='', **kwargs):
        """Create an IO port.

        name is the port name, as returned by ioport_names().
        """
        BaseInput.__init__(self, name, **kwargs)
        BaseOutput.__init__(self, name, **kwargs)