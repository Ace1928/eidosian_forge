import atexit
import os
import sys
import traceback
import eventlet
from eventlet import event, greenio, greenthread, patcher, timeout

    a simple proxy-wrapper of any object that comes with a
    methods-only interface, in order to forward every method
    invocation onto a thread in the native-thread pool.  A key
    restriction is that the object's methods should not switch
    greenlets or use Eventlet primitives, since they are in a
    different thread from the main hub, and therefore might behave
    unexpectedly.  This is for running native-threaded code
    only.

    It's common to want to have some of the attributes or return
    values also wrapped in Proxy objects (for example, database
    connection objects produce cursor objects which also should be
    wrapped in Proxy objects to remain nonblocking).  *autowrap*, if
    supplied, is a collection of types; if an attribute or return
    value matches one of those types (via isinstance), it will be
    wrapped in a Proxy.  *autowrap_names* is a collection
    of strings, which represent the names of attributes that should be
    wrapped in Proxy objects when accessed.
    