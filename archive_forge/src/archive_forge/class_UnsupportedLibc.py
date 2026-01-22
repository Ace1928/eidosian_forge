import os
import sys
import threading
from wandb_watchdog.utils import platform
from wandb_watchdog.utils.compat import Event
class UnsupportedLibc(Exception):
    pass