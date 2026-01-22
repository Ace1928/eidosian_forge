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
class GatewayKernelSpecManager(KernelSpecManager):
    """A gateway kernel spec manager."""

    def __init__(self, **kwargs):
        """Initialize a gateway kernel spec manager."""
        super().__init__(**kwargs)
        base_endpoint = url_path_join(GatewayClient.instance().url or '', GatewayClient.instance().kernelspecs_endpoint)
        self.base_endpoint = GatewayKernelSpecManager._get_endpoint_for_user_filter(base_endpoint)
        self.base_resource_endpoint = url_path_join(GatewayClient.instance().url or '', GatewayClient.instance().kernelspecs_resource_endpoint)

    @staticmethod
    def _get_endpoint_for_user_filter(default_endpoint):
        """Get the endpoint for a user filter."""
        kernel_user = os.environ.get('KERNEL_USERNAME')
        if kernel_user:
            return '?user='.join([default_endpoint, kernel_user])
        return default_endpoint

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

    def _get_kernelspecs_endpoint_url(self, kernel_name=None):
        """Builds a url for the kernels endpoint
        Parameters
        ----------
        kernel_name : kernel name (optional)
        """
        if kernel_name:
            return url_path_join(self.base_endpoint, url_escape(kernel_name))
        return self.base_endpoint

    async def get_all_specs(self):
        """Get all of the kernel specs for the gateway."""
        fetched_kspecs = await self.list_kernel_specs()
        if not self.parent:
            return {}
        km = self.parent.kernel_manager
        remote_default_kernel_name = fetched_kspecs.get('default')
        if remote_default_kernel_name != km.default_kernel_name:
            self.log.info(f"Default kernel name on Gateway server ({remote_default_kernel_name}) differs from Notebook server ({km.default_kernel_name}).  Updating to Gateway server's value.")
            km.default_kernel_name = remote_default_kernel_name
        remote_kspecs = fetched_kspecs.get('kernelspecs')
        return remote_kspecs

    async def list_kernel_specs(self):
        """Get a list of kernel specs."""
        kernel_spec_url = self._get_kernelspecs_endpoint_url()
        self.log.debug(f'Request list kernel specs at: {kernel_spec_url}')
        response = await gateway_request(kernel_spec_url, method='GET')
        kernel_specs = json_decode(response.body)
        kernel_specs = self._replace_path_kernelspec_resources(kernel_specs)
        return kernel_specs

    async def get_kernel_spec(self, kernel_name, **kwargs):
        """Get kernel spec for kernel_name.

        Parameters
        ----------
        kernel_name : str
            The name of the kernel.
        """
        kernel_spec_url = self._get_kernelspecs_endpoint_url(kernel_name=str(kernel_name))
        self.log.debug(f'Request kernel spec at: {kernel_spec_url}')
        try:
            response = await gateway_request(kernel_spec_url, method='GET')
        except web.HTTPError as error:
            if error.status_code == 404:
                msg = f'kernelspec {kernel_name} not found on Gateway server at: {GatewayClient.instance().url}'
                raise KeyError(msg) from None
            else:
                raise
        else:
            kernel_spec = json_decode(response.body)
        return kernel_spec

    async def get_kernel_spec_resource(self, kernel_name, path):
        """Get kernel spec for kernel_name.

        Parameters
        ----------
        kernel_name : str
            The name of the kernel.
        path : str
            The name of the desired resource
        """
        kernel_spec_resource_url = url_path_join(self.base_resource_endpoint, str(kernel_name), str(path))
        self.log.debug(f"Request kernel spec resource '{path}' at: {kernel_spec_resource_url}")
        try:
            response = await gateway_request(kernel_spec_resource_url, method='GET')
        except web.HTTPError as error:
            if error.status_code == 404:
                kernel_spec_resource = None
            else:
                raise
        else:
            kernel_spec_resource = response.body
        return kernel_spec_resource