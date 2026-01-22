import zmq
from zmq.devices.basedevice import Device, ProcessDevice, ThreadDevice
class ProcessProxy(ProxyBase, ProcessDevice):
    """Proxy in a Process. See Proxy for more."""