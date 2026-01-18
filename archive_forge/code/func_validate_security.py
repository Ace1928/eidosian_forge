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
def validate_security(self, app: t.Any, ssl_options: dict[str, t.Any] | None=None) -> None:
    """Validate security."""
    if self.password_required and (not self.hashed_password):
        self.log.critical(_i18n('Jupyter servers are configured to only be run with a password.'))
        self.log.critical(_i18n('Hint: run the following command to set a password'))
        self.log.critical(_i18n('\t$ python -m jupyter_server.auth password'))
        sys.exit(1)
    self.login_handler_class.validate_security(app, ssl_options)