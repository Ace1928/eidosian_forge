import datetime
import json
import os
import socket
from tempfile import NamedTemporaryFile
import threading
import time
import sys
import google.auth
from google.auth import _helpers
from googleapiclient import discovery
from six.moves import BaseHTTPServer
from google.oauth2 import service_account
import pytest
from mock import patch
@staticmethod
def _find_open_port():
    s = socket.socket()
    s.bind(('', 0))
    return s.getsockname()[1]