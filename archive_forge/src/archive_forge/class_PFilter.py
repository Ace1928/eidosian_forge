import sys
import struct
import traceback
import threading
import logging
from paramiko.common import (
from paramiko.config import SSHConfig
class PFilter:

    def filter(self, record):
        record._threadid = get_thread_id()
        return True