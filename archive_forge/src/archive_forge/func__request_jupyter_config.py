import asyncio
import io
import inspect
import logging
import os
import queue
import uuid
import sys
import threading
import time
from typing_extensions import Literal
from werkzeug.serving import make_server
def _request_jupyter_config(timeout=2):
    if _dash_comm.kernel is None:
        return
    _send_jupyter_config_comm_request()
    shell = get_ipython()
    kernel = shell.kernel
    captured_events = []

    def capture_event(stream, ident, parent):
        captured_events.append((stream, ident, parent))
    kernel.shell_handlers['execute_request'] = capture_event
    shell.execution_count += 1
    t0 = time.time()
    while True:
        if time.time() - t0 > timeout:
            raise EnvironmentError('Unable to communicate with the jupyter_dash notebook or JupyterLab \nextension required to infer Jupyter configuration.')
        if _jupyter_comm_response_received():
            break
        if asyncio.iscoroutinefunction(kernel.do_one_iteration):
            loop = asyncio.get_event_loop()
            nest_asyncio.apply(loop)
            loop.run_until_complete(kernel.do_one_iteration())
        else:
            kernel.do_one_iteration()
    kernel.shell_handlers['execute_request'] = kernel.execute_request
    sys.stdout.flush()
    sys.stderr.flush()
    for stream, ident, parent in captured_events:
        kernel.set_parent(ident, parent)
        kernel.execute_request(stream, ident, parent)