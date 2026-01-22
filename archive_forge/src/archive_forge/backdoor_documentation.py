from code import InteractiveConsole
import errno
import socket
import sys
import eventlet
from eventlet import hubs
from eventlet.support import greenlets, get_errno
Sets up an interactive console on a socket with a single connected
    client.  This does not block the caller, as it spawns a new greenlet to
    handle the console.  This is meant to be called from within an accept loop
    (such as backdoor_server).
    