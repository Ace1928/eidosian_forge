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
def _auth_required(self, auth, msg):
    if not auth:
        auth = self.auth
    if not auth:
        msg_fmt = 'An auth plugin is required to %s'
        raise exceptions.MissingAuthPlugin(msg_fmt % msg)
    return auth