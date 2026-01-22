import json
import os
from typing import Any, Dict, List
from jupyter_core.utils import ensure_async
from tornado import web
from jupyter_server._tz import isoformat, utcfromtimestamp
from jupyter_server.auth.decorator import authorized
from ...base.handlers import APIHandler, JupyterHandler
class IdentityHandler(APIHandler):
    """Get the current user's identity model"""

    @web.authenticated
    async def get(self):
        """Get the identity model."""
        permissions_json: str = self.get_argument('permissions', '')
        bad_permissions_msg = f'permissions should be a JSON dict of {{"resource": ["action",]}}, got {permissions_json!r}'
        if permissions_json:
            try:
                permissions_to_check = json.loads(permissions_json)
            except ValueError as e:
                raise web.HTTPError(400, bad_permissions_msg) from e
            if not isinstance(permissions_to_check, dict):
                raise web.HTTPError(400, bad_permissions_msg)
        else:
            permissions_to_check = {}
        permissions: Dict[str, List[str]] = {}
        user = self.current_user
        for resource, actions in permissions_to_check.items():
            if not isinstance(resource, str) or not isinstance(actions, list) or (not all((isinstance(action, str) for action in actions))):
                raise web.HTTPError(400, bad_permissions_msg)
            allowed = permissions[resource] = []
            for action in actions:
                authorized = await ensure_async(self.authorizer.is_authorized(self, user, action, resource))
                if authorized:
                    allowed.append(action)
        identity: Dict[str, Any] = self.identity_provider.identity_model(user)
        model = {'identity': identity, 'permissions': permissions}
        self.write(json.dumps(model))