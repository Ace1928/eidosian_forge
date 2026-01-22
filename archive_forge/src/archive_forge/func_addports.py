from suds import *
import suds.metrics as metrics
from suds.sax import Namespace
from logging import getLogger
def addports(self):
    """
        Look through the list of service ports and construct a list of tuples
        where each tuple is used to describe a port and its list of methods as:
        (port, [method]).  Each method is a tuple: (name, [pdef,..]) where each
        pdef is a tuple: (param-name, type).
        """
    timer = metrics.Timer()
    timer.start()
    for port in self.service.ports:
        p = self.findport(port)
        for op in list(port.binding.operations.values()):
            m = p[0].method(op.name)
            binding = m.binding.input
            method = (m.name, binding.param_defs(m))
            p[1].append(method)
            metrics.log.debug("method '%s' created: %s", m.name, timer)
        p[1].sort()
    timer.stop()