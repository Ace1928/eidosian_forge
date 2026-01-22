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
def _send_jupyter_config_comm_request():
    if get_ipython() is not None:
        if _dash_comm.kernel is not None:
            _caller['parent'] = _dash_comm.kernel.get_parent()
            _dash_comm.send({'type': 'base_url_request'})