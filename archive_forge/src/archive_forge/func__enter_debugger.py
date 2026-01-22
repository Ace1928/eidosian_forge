import logging
import signal
import sys
import threading
import traceback
from pdb import Pdb
from scrapy.utils.engine import format_engine_status
from scrapy.utils.trackref import format_live_refs
def _enter_debugger(self, signum, frame):
    Pdb().set_trace(frame.f_back)