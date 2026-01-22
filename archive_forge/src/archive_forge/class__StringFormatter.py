import datetime
import functools
import hashlib
import json
import logging
import os
import platform
import socket
import sys
import time
import urllib
import uuid
import requests
import keystoneauth1
from keystoneauth1 import _utils as utils
from keystoneauth1 import discover
from keystoneauth1 import exceptions
class _StringFormatter(object):
    """A String formatter that fetches values on demand."""

    def __init__(self, session, auth):
        self.session = session
        self.auth = auth

    def __getitem__(self, item):
        if item == 'project_id':
            value = self.session.get_project_id(self.auth)
        elif item == 'user_id':
            value = self.session.get_user_id(self.auth)
        else:
            raise AttributeError(item)
        if not value:
            raise ValueError('This type of authentication does not provide a %s that can be substituted' % item)
        return value