from __future__ import (absolute_import, division, print_function)
from functools import wraps
from os import environ
from os import path
from datetime import datetime

    Manage the locking of a snapshot. Check for bad lock times.
    See check_snapshot_lock_options() which has additional checks.
    