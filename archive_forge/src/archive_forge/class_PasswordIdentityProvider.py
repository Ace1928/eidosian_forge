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
class PasswordIdentityProvider(IdentityProvider):
    """A password identity provider."""
    hashed_password = Unicode('', config=True, help=_i18n('\n            Hashed password to use for web authentication.\n\n            To generate, type in a python/IPython shell:\n\n                from jupyter_server.auth import passwd; passwd()\n\n            The string should be of the form type:salt:hashed-password.\n            '))
    password_required = Bool(False, config=True, help=_i18n("\n            Forces users to use a password for the Jupyter server.\n            This is useful in a multi user environment, for instance when\n            everybody in the LAN can access each other's machine through ssh.\n\n            In such a case, serving on localhost is not secure since\n            any user can connect to the Jupyter server via ssh.\n\n            "))
    allow_password_change = Bool(True, config=True, help=_i18n('\n            Allow password to be changed at login for the Jupyter server.\n\n            While logging in with a token, the Jupyter server UI will give the opportunity to\n            the user to enter a new password at the same time that will replace\n            the token login mechanism.\n\n            This can be set to False to prevent changing password from the UI/API.\n            '))

    @default('need_token')
    def _need_token_default(self):
        return not bool(self.hashed_password)

    @property
    def login_available(self) -> bool:
        """Whether a LoginHandler is needed - and therefore whether the login page should be displayed."""
        return self.auth_enabled

    @property
    def auth_enabled(self) -> bool:
        """Return whether any auth is enabled"""
        return bool(self.hashed_password or self.token)

    def passwd_check(self, password):
        """Check password against our stored hashed password"""
        return passwd_check(self.hashed_password, password)

    def process_login_form(self, handler: web.RequestHandler) -> User | None:
        """Process login form data

        Return authenticated User if successful, None if not.
        """
        typed_password = handler.get_argument('password', default='')
        new_password = handler.get_argument('new_password', default='')
        user = None
        if not self.auth_enabled:
            self.log.warning('Accepting anonymous login because auth fully disabled!')
            return self.generate_anonymous_user(handler)
        if self.passwd_check(typed_password) and (not new_password):
            return self.generate_anonymous_user(handler)
        elif self.token and self.token == typed_password:
            user = self.generate_anonymous_user(handler)
            if new_password and self.allow_password_change:
                config_dir = handler.settings.get('config_dir', '')
                config_file = os.path.join(config_dir, 'jupyter_server_config.json')
                self.hashed_password = set_password(new_password, config_file=config_file)
                self.log.info(_i18n(f'Wrote hashed password to {config_file}'))
        return user

    def validate_security(self, app: t.Any, ssl_options: dict[str, t.Any] | None=None) -> None:
        """Handle security validation."""
        super().validate_security(app, ssl_options)
        if self.password_required and (not self.hashed_password):
            self.log.critical(_i18n('Jupyter servers are configured to only be run with a password.'))
            self.log.critical(_i18n('Hint: run the following command to set a password'))
            self.log.critical(_i18n('\t$ python -m jupyter_server.auth password'))
            sys.exit(1)