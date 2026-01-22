from typing import Any, Optional
from jupyter_server.auth.identity import IdentityProvider
from jupyter_server.auth.login import LoginFormHandler
class VoilaIdentityProvider(IdentityProvider):

    @property
    def auth_enabled(self) -> bool:
        """Return whether any auth is enabled"""
        return bool(self.token)