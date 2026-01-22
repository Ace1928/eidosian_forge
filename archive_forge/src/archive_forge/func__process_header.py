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
@staticmethod
def _process_header(header):
    """Redact the secure headers to be logged."""
    secure_headers = ('authorization', 'x-auth-token', 'x-subject-token', 'x-service-token')
    if header[0].lower() in secure_headers:
        token_hasher = hashlib.sha256()
        token_hasher.update(header[1].encode('utf-8'))
        token_hash = token_hasher.hexdigest()
        return (header[0], '{SHA256}%s' % token_hash)
    return header