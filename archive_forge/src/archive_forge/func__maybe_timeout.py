import socket
import trio
import trio.socket  # type: ignore
import dns._asyncbackend
import dns._features
import dns.exception
import dns.inet
def _maybe_timeout(timeout):
    if timeout is not None:
        return trio.move_on_after(timeout)
    else:
        return dns._asyncbackend.NullContext()