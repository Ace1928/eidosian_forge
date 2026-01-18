import threading
import sys
from paste.util import filemixin

threadedprint.py
================

:author: Ian Bicking
:date: 12 Jul 2004

Multi-threaded printing; allows the output produced via print to be
separated according to the thread.

To use this, you must install the catcher, like::

    threadedprint.install()

The installation optionally takes one of three parameters:

default
    The default destination for print statements (e.g., ``sys.stdout``).
factory
    A function that will produce the stream for a thread, given the
    thread's name.
paramwriter
    Instead of writing to a file-like stream, this function will be
    called like ``paramwriter(thread_name, text)`` for every write.

The thread name is the value returned by
``threading.current_thread().name``, a string (typically something
like Thread-N).

You can also submit file-like objects for specific threads, which will
override any of these parameters.  To do this, call ``register(stream,
[threadName])``.  ``threadName`` is optional, and if not provided the
stream will be registered for the current thread.

If no specific stream is registered for a thread, and no default has
been provided, then an error will occur when anything is written to
``sys.stdout`` (or printed).

Note: the stream's ``write`` method will be called in the thread the
text came from, so you should consider thread safety, especially if
multiple threads share the same writer.

Note: if you want access to the original standard out, use
``sys.__stdout__``.

You may also uninstall this, via::

    threadedprint.uninstall()

TODO
----

* Something with ``sys.stderr``.
* Some default handlers.  Maybe something that hooks into `logging`.
* Possibly cache the results of ``factory`` calls.  This would be a
  semantic change.

