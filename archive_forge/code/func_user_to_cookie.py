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
def user_to_cookie(self, user: User) -> str:
    """Serialize a user to a string for storage in a cookie

        If overriding in a subclass, make sure to define user_from_cookie as well.

        Default is just the user's username.
        """
    cookie = json.dumps({'username': user.username, 'name': user.name, 'display_name': user.display_name, 'initials': user.initials, 'color': user.color})
    return cookie