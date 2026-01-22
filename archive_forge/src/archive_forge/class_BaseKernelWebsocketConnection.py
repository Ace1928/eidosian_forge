import json
import struct
from typing import Any, List
from jupyter_client.session import Session
from tornado.websocket import WebSocketHandler
from traitlets import Float, Instance, Unicode, default
from traitlets.config import LoggingConfigurable
from jupyter_client.jsonutil import extract_dates
from jupyter_server.transutils import _i18n
from .abc import KernelWebsocketConnectionABC
class BaseKernelWebsocketConnection(LoggingConfigurable):
    """A configurable base class for connecting Kernel WebSockets to ZMQ sockets."""
    kernel_ws_protocol = Unicode(None, allow_none=True, config=True, help=_i18n('Preferred kernel message protocol over websocket to use (default: None). If an empty string is passed, select the legacy protocol. If None, the selected protocol will depend on what the front-end supports (usually the most recent protocol supported by the back-end and the front-end).'))

    @property
    def kernel_manager(self):
        """The kernel manager."""
        return self.parent

    @property
    def multi_kernel_manager(self):
        """The multi kernel manager."""
        return self.kernel_manager.parent

    @property
    def kernel_id(self):
        """The kernel id."""
        return self.kernel_manager.kernel_id

    @property
    def session_id(self):
        """The session id."""
        return self.session.session
    kernel_info_timeout = Float()

    @default('kernel_info_timeout')
    def _default_kernel_info_timeout(self):
        return self.multi_kernel_manager.kernel_info_timeout
    session = Instance(klass=Session, config=True)

    @default('session')
    def _default_session(self):
        return Session(config=self.config)
    websocket_handler = Instance(WebSocketHandler)

    async def connect(self):
        """Handle a connect."""
        raise NotImplementedError()

    async def disconnect(self):
        """Handle a disconnect."""
        raise NotImplementedError()

    def handle_incoming_message(self, incoming_msg: str) -> None:
        """Handle an incoming message."""
        raise NotImplementedError()

    def handle_outgoing_message(self, stream: str, outgoing_msg: List[Any]) -> None:
        """Handle an outgoing message."""
        raise NotImplementedError()