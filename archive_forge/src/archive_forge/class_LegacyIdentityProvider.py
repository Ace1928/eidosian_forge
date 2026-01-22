from __future__ import annotations
import binascii
import datetime
import json
import os
import re
import sys
import typing as t
import uuid
from dataclasses import asdict, dataclass
from http.cookies import Morsel
from tornado import escape, httputil, web
from traitlets import Bool, Dict, Type, Unicode, default
from traitlets.config import LoggingConfigurable
from jupyter_server.transutils import _i18n
from .security import passwd_check, set_password
from .utils import get_anonymous_username
class LegacyIdentityProvider(PasswordIdentityProvider):
    """Legacy IdentityProvider for use with custom LoginHandlers

    Login configuration has moved from LoginHandler to IdentityProvider
    in Jupyter Server 2.0.
    """
    settings = Dict()

    @default('settings')
    def _default_settings(self):
        return {'token': self.token, 'password': self.hashed_password}

    @default('login_handler_class')
    def _default_login_handler_class(self):
        from .login import LegacyLoginHandler
        return LegacyLoginHandler

    @property
    def auth_enabled(self):
        return self.login_available

    def get_user(self, handler: web.RequestHandler) -> User | None:
        """Get the user."""
        user = self.login_handler_class.get_user(handler)
        if user is None:
            return None
        return _backward_compat_user(user)

    @property
    def login_available(self) -> bool:
        return bool(self.login_handler_class.get_login_available(self.settings))

    def should_check_origin(self, handler: web.RequestHandler) -> bool:
        """Whether we should check origin."""
        return bool(self.login_handler_class.should_check_origin(handler))

    def is_token_authenticated(self, handler: web.RequestHandler) -> bool:
        """Whether we are token authenticated."""
        return bool(self.login_handler_class.is_token_authenticated(handler))

    def validate_security(self, app: t.Any, ssl_options: dict[str, t.Any] | None=None) -> None:
        """Validate security."""
        if self.password_required and (not self.hashed_password):
            self.log.critical(_i18n('Jupyter servers are configured to only be run with a password.'))
            self.log.critical(_i18n('Hint: run the following command to set a password'))
            self.log.critical(_i18n('\t$ python -m jupyter_server.auth password'))
            sys.exit(1)
        self.login_handler_class.validate_security(app, ssl_options)