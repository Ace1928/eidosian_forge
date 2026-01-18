from io import BytesIO
from twisted.internet import protocol
A Protocol that stores state for you.

    state is a pair (function, num_bytes). When num_bytes bytes of data arrives
    from the network, function is called. It is expected to return the next
    state or None to keep same state. Initial state is returned by
    getInitialState (override it).
    