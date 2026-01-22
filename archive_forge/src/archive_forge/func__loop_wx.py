import os
import platform
import sys
from functools import partial
import zmq
from packaging.version import Version as V
from traitlets.config.application import Application
def _loop_wx(app):
    """Inner-loop for running the Wx eventloop

    Pulled from guisupport.start_event_loop in IPython < 5.2,
    since IPython 5.2 only checks `get_ipython().active_eventloop` is defined,
    rather than if the eventloop is actually running.
    """
    app._in_event_loop = True
    app.MainLoop()
    app._in_event_loop = False