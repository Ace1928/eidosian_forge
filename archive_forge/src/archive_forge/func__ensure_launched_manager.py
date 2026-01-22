import importlib.machinery
import logging
import multiprocessing
import os
import queue
import sys
import threading
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union, cast
import wandb
from wandb.sdk.interface.interface import InterfaceBase
from wandb.sdk.interface.interface_queue import InterfaceQueue
from wandb.sdk.internal.internal import wandb_internal
from wandb.sdk.internal.settings_static import SettingsStatic
from wandb.sdk.lib.mailbox import Mailbox
from wandb.sdk.wandb_manager import _Manager
from wandb.sdk.wandb_settings import Settings
def _ensure_launched_manager(self) -> None:
    assert self._manager
    svc = self._manager._get_service()
    assert svc
    svc_iface = svc.service_interface
    svc_transport = svc_iface.get_transport()
    if svc_transport == 'tcp':
        from ..interface.interface_sock import InterfaceSock
        svc_iface_sock = cast('ServiceSockInterface', svc_iface)
        sock_client = svc_iface_sock._get_sock_client()
        sock_interface = InterfaceSock(sock_client, mailbox=self._mailbox)
        self.interface = sock_interface
    else:
        raise AssertionError(f'Unsupported service transport: {svc_transport}')