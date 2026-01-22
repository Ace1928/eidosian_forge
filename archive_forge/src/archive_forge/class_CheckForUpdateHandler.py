import abc
import hashlib
import json
import xml.etree.ElementTree as ET  # noqa
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Optional, Tuple, Union
from jupyter_server.base.handlers import APIHandler
from jupyterlab_server.translation_utils import translator
from packaging.version import parse
from tornado import httpclient, web
from jupyterlab._version import __version__
class CheckForUpdateHandler(APIHandler):
    """Check for Updates API handler.

    Args:
        update_check: The class checking for a new version
    """

    def initialize(self, update_checker: Optional[CheckForUpdate]=None) -> None:
        super().initialize()
        self.update_checker = NeverCheckForUpdate(__version__) if update_checker is None else update_checker
        self.update_checker.logger = self.log

    @web.authenticated
    async def get(self):
        """Check for updates.
        Response:
            {
                "notification": Optional[Notification]
            }
        """
        notification = None
        out = await self.update_checker()
        if out:
            message, link = (out, ()) if isinstance(out, str) else out
            now = datetime.now(tz=timezone.utc).timestamp() * 1000.0
            hash_ = hashlib.sha1(message.encode()).hexdigest()
            notification = Notification(message=message, createdAt=now, modifiedAt=now, type='info', link=link, options={'data': {'id': hash_, 'tags': ['update']}})
        self.set_status(200)
        self.finish(json.dumps({'notification': None if notification is None else asdict(notification)}))