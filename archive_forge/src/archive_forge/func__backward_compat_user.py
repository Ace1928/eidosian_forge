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
def _backward_compat_user(got_user: t.Any) -> User:
    """Backward-compatibility for LoginHandler.get_user

    Prior to 2.0, LoginHandler.get_user could return anything truthy.

    Typically, this was either a simple string username,
    or a simple dict.

    Make some effort to allow common patterns to keep working.
    """
    if isinstance(got_user, str):
        return User(username=got_user)
    elif isinstance(got_user, dict):
        kwargs = {}
        if 'username' not in got_user and 'name' in got_user:
            kwargs['username'] = got_user['name']
        for field in User.__dataclass_fields__:
            if field in got_user:
                kwargs[field] = got_user[field]
        try:
            return User(**kwargs)
        except TypeError:
            msg = f'Unrecognized user: {got_user}'
            raise ValueError(msg) from None
    else:
        msg = f'Unrecognized user: {got_user}'
        raise ValueError(msg)