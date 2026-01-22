from zmq import PUB
from zmq.devices.monitoredqueue import monitored_queue
from zmq.devices.proxydevice import ProcessProxy, Proxy, ProxyBase, ThreadProxy
class MonitoredQueueBase(ProxyBase):
    """Base class for overriding methods."""
    _in_prefix = b''
    _out_prefix = b''

    def __init__(self, in_type, out_type, mon_type=PUB, in_prefix=b'in', out_prefix=b'out'):
        ProxyBase.__init__(self, in_type=in_type, out_type=out_type, mon_type=mon_type)
        self._in_prefix = in_prefix
        self._out_prefix = out_prefix

    def run_device(self):
        ins, outs, mons = self._setup_sockets()
        monitored_queue(ins, outs, mons, self._in_prefix, self._out_prefix)