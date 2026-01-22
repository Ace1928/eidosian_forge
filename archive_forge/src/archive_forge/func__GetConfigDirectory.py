from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import base64
import errno
import io
import json
import logging
import os
import subprocess
from containerregistry.client import docker_name
import httplib2
from oauth2client import client as oauth2client
import six
def _GetConfigDirectory():
    if os.environ.get('DOCKER_CONFIG') is not None:
        return os.environ.get('DOCKER_CONFIG')
    else:
        return os.path.join(_GetUserHomeDir(), '.docker')