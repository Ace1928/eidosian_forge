import os
import socket
import socketserver
import sys
import time
import paramiko
from .. import osutils, trace, urlutils
from ..transport import ssh
from . import test_server
class FakeChannel:

    def get_transport(self):
        return self

    def get_log_channel(self):
        return 'brz.paramiko'

    def get_name(self):
        return '1'

    def get_hexdump(self):
        return False

    def close(self):
        pass