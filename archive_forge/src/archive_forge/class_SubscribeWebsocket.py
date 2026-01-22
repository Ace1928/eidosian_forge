from __future__ import annotations
import json
from datetime import datetime
from typing import Any, Dict, Optional, cast
import jupyter_events.logger
from jupyter_core.utils import ensure_async
from tornado import web, websocket
from jupyter_server.auth.decorator import authorized, ws_authenticated
from jupyter_server.base.handlers import JupyterHandler
from ...base.handlers import APIHandler
class SubscribeWebsocket(JupyterHandler, websocket.WebSocketHandler):
    """Websocket handler for subscribing to events"""
    auth_resource = AUTH_RESOURCE

    async def pre_get(self):
        """Handles authorization when
        attempting to subscribe to events emitted by
        Jupyter Server's eventbus.
        """
        user = self.current_user
        authorized = await ensure_async(self.authorizer.is_authorized(self, user, 'execute', 'events'))
        if not authorized:
            raise web.HTTPError(403)

    @ws_authenticated
    async def get(self, *args, **kwargs):
        """Get an event socket."""
        await ensure_async(self.pre_get())
        res = super().get(*args, **kwargs)
        if res is not None:
            await res

    async def event_listener(self, logger: jupyter_events.logger.EventLogger, schema_id: str, data: dict[str, Any]) -> None:
        """Write an event message."""
        capsule = dict(schema_id=schema_id, **data)
        self.write_message(json.dumps(capsule))

    def open(self):
        """Routes events that are emitted by Jupyter Server's
        EventBus to a WebSocket client in the browser.
        """
        self.event_logger.add_listener(listener=self.event_listener)

    def on_close(self):
        """Handle a socket close."""
        self.event_logger.remove_listener(listener=self.event_listener)