import io
import os
import signal
import sys
import threading
import warnings
from ._ext import _billiard
from . import spawn
from . import util
from .compat import spawnv_passfds
Unregister name of semaphore with semaphore tracker.