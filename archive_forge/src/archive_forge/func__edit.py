import abc
import contextlib
import hashlib
import os
from urllib import parse
from troveclient.apiclient import base
from troveclient.apiclient import exceptions
from troveclient import common
from troveclient import utils
def _edit(self, url, body):
    resp, body = self.api.client.patch(url, body=body)
    return body