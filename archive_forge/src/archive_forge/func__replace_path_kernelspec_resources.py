from __future__ import annotations
import asyncio
import datetime
import json
import os
from logging import Logger
from queue import Empty, Queue
from threading import Thread
from time import monotonic
from typing import Any, Optional, cast
import websocket
from jupyter_client.asynchronous.client import AsyncKernelClient
from jupyter_client.clientabc import KernelClientABC
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client.managerabc import KernelManagerABC
from jupyter_core.utils import ensure_async
from tornado import web
from tornado.escape import json_decode, json_encode, url_escape, utf8
from traitlets import DottedObjectName, Instance, Type, default
from .._tz import UTC, utcnow
from ..services.kernels.kernelmanager import (
from ..services.sessions.sessionmanager import SessionManager
from ..utils import url_path_join
from .gateway_client import GatewayClient, gateway_request
def _replace_path_kernelspec_resources(self, kernel_specs):
    """Helper method that replaces any gateway base_url with the server's base_url
        This enables clients to properly route through jupyter_server to a gateway
        for kernel resources such as logo files
        """
    if not self.parent:
        return {}
    kernelspecs = kernel_specs['kernelspecs']
    for kernel_name in kernelspecs:
        resources = kernelspecs[kernel_name]['resources']
        for resource_name in resources:
            original_path = resources[resource_name]
            split_eg_base_url = str.rsplit(original_path, sep='/kernelspecs/', maxsplit=1)
            if len(split_eg_base_url) > 1:
                new_path = url_path_join(self.parent.base_url, 'kernelspecs', split_eg_base_url[1])
                kernel_specs['kernelspecs'][kernel_name]['resources'][resource_name] = new_path
                if original_path != new_path:
                    self.log.debug(f'Replaced original kernel resource path {original_path} with new path {kernel_specs['kernelspecs'][kernel_name]['resources'][resource_name]}')
    return kernel_specs