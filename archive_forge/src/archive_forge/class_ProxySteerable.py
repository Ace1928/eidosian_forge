import zmq
from zmq.devices.proxydevice import ProcessProxy, Proxy, ThreadProxy
class ProxySteerable(ProxySteerableBase, Proxy):
    """Class for running a steerable proxy in the background.

    See zmq.devices.Proxy for most of the spec.  If the control socket is not
    NULL, the proxy supports control flow, provided by the socket.

    If PAUSE is received on this socket, the proxy suspends its activities. If
    RESUME is received, it goes on. If TERMINATE is received, it terminates
    smoothly.  If the control socket is NULL, the proxy behave exactly as if
    zmq.devices.Proxy had been used.

    This subclass adds a <method>_ctrl version of each <method>_{in|out}
    method, for configuring the control socket.

    .. versionadded:: libzmq-4.1
    .. versionadded:: 18.0
    """