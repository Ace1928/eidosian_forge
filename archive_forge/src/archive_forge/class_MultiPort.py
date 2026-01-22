import random
import threading
import time
from .messages import Message
from .parser import Parser
class MultiPort(BaseIOPort):

    def __init__(self, ports, yield_ports=False):
        BaseIOPort.__init__(self, 'multi')
        self.ports = list(ports)
        self.yield_ports = yield_ports

    def _send(self, message):
        for port in self.ports:
            if not port.closed:
                port.send(message)

    def _receive(self, block=True):
        self._messages.extend(multi_receive(self.ports, yield_ports=self.yield_ports, block=block))