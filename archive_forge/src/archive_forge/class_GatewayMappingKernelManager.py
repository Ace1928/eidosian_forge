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
class GatewayMappingKernelManager(AsyncMappingKernelManager):
    """Kernel manager that supports remote kernels hosted by Jupyter Kernel or Enterprise Gateway."""
    _kernels: dict[str, GatewayKernelManager] = {}

    @default('kernel_manager_class')
    def _default_kernel_manager_class(self):
        return 'jupyter_server.gateway.managers.GatewayKernelManager'

    @default('shared_context')
    def _default_shared_context(self):
        return False

    def __init__(self, **kwargs):
        """Initialize a gateway mapping kernel manager."""
        super().__init__(**kwargs)
        self.kernels_url = url_path_join(GatewayClient.instance().url or '', GatewayClient.instance().kernels_endpoint or '')

    def remove_kernel(self, kernel_id):
        """Complete override since we want to be more tolerant of missing keys"""
        try:
            return self._kernels.pop(kernel_id)
        except KeyError:
            pass

    async def start_kernel(self, *, kernel_id=None, path=None, **kwargs):
        """Start a kernel for a session and return its kernel_id.

        Parameters
        ----------
        kernel_id : uuid
            The uuid to associate the new kernel with. If this
            is not None, this kernel will be persistent whenever it is
            requested.
        path : API path
            The API path (unicode, '/' delimited) for the cwd.
            Will be transformed to an OS path relative to root_dir.
        """
        self.log.info(f"Request start kernel: kernel_id={kernel_id}, path='{path}'")
        if kernel_id is None and path is not None:
            kwargs['cwd'] = self.cwd_for_path(path)
        km = self.kernel_manager_factory(parent=self, log=self.log)
        await km.start_kernel(kernel_id=kernel_id, **kwargs)
        kernel_id = km.kernel_id
        self._kernels[kernel_id] = km
        if not self._initialized_culler:
            self.initialize_culler()
        return kernel_id

    async def kernel_model(self, kernel_id):
        """Return a dictionary of kernel information described in the
        JSON standard model.

        Parameters
        ----------
        kernel_id : uuid
            The uuid of the kernel.
        """
        model = None
        km = self.get_kernel(str(kernel_id))
        if km:
            model = km.kernel
        return model

    async def list_kernels(self, **kwargs):
        """Get a list of running kernels from the Gateway server.

        We'll use this opportunity to refresh the models in each of
        the kernels we're managing.
        """
        self.log.debug(f'Request list kernels: {self.kernels_url}')
        response = await gateway_request(self.kernels_url, method='GET')
        kernels = json_decode(response.body)
        kernel_models = {}
        for model in kernels:
            kid = model['id']
            if kid in self._kernels:
                await self._kernels[kid].refresh_model(model)
                kernel_models[kid] = model
        our_kernels = self._kernels.copy()
        culled_ids = []
        for kid, _ in our_kernels.items():
            if kid not in kernel_models:
                self.log.warning(f'Kernel {kid} not present in the list of kernels - possibly culled on Gateway server.')
                try:
                    model = await self._kernels[kid].refresh_model()
                except web.HTTPError:
                    model = None
                if model:
                    kernel_models[kid] = model
                else:
                    self.log.warning(f'Kernel {kid} no longer active - probably culled on Gateway server.')
                    self._kernels.pop(kid, None)
                    culled_ids.append(kid)
        return list(kernel_models.values())

    async def shutdown_kernel(self, kernel_id, now=False, restart=False):
        """Shutdown a kernel by its kernel uuid.

        Parameters
        ==========
        kernel_id : uuid
            The id of the kernel to shutdown.
        now : bool
            Shutdown the kernel immediately (True) or gracefully (False)
        restart : bool
            The purpose of this shutdown is to restart the kernel (True)
        """
        km = self.get_kernel(kernel_id)
        await ensure_async(km.shutdown_kernel(now=now, restart=restart))
        self.remove_kernel(kernel_id)

    async def restart_kernel(self, kernel_id, now=False, **kwargs):
        """Restart a kernel by its kernel uuid.

        Parameters
        ==========
        kernel_id : uuid
            The id of the kernel to restart.
        """
        km = self.get_kernel(kernel_id)
        await ensure_async(km.restart_kernel(now=now, **kwargs))

    async def interrupt_kernel(self, kernel_id, **kwargs):
        """Interrupt a kernel by its kernel uuid.

        Parameters
        ==========
        kernel_id : uuid
            The id of the kernel to interrupt.
        """
        km = self.get_kernel(kernel_id)
        await ensure_async(km.interrupt_kernel())

    async def shutdown_all(self, now=False):
        """Shutdown all kernels."""
        kids = list(self._kernels)
        for kernel_id in kids:
            km = self.get_kernel(kernel_id)
            await ensure_async(km.shutdown_kernel(now=now))
            self.remove_kernel(kernel_id)

    async def cull_kernels(self):
        """Override cull_kernels, so we can be sure their state is current."""
        await self.list_kernels()
        await super().cull_kernels()