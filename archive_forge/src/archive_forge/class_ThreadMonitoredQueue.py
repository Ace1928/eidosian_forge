from zmq import PUB
from zmq.devices.monitoredqueue import monitored_queue
from zmq.devices.proxydevice import ProcessProxy, Proxy, ProxyBase, ThreadProxy
class ThreadMonitoredQueue(MonitoredQueueBase, ThreadProxy):
    """Run zmq.monitored_queue in a background thread.

    See MonitoredQueue and Proxy for details.
    """