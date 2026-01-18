import asyncore
import binascii
import collections
import errno
import functools
import hashlib
import hmac
import math
import os
import pickle
import socket
import struct
import time
import futurist
from oslo_utils import excutils
from taskflow.engines.action_engine import executor as base
from taskflow import logging
from taskflow import task as ta
from taskflow.types import notifier as nt
from taskflow.utils import iter_utils
from taskflow.utils import misc
from taskflow.utils import schema_utils as su
from taskflow.utils import threading_utils
Submit a function to run the given task (with given args/kwargs).

        NOTE(harlowja): Adjust all events to be proxies instead since we want
        those callbacks to be activated in this process, not in the child,
        also since typically callbacks are functors (or callables) we can
        not pickle those in the first place...

        To make sure people understand how this works, the following is a
        lengthy description of what is going on here, read at will:

        So to ensure that we are proxying task triggered events that occur
        in the executed subprocess (which will be created and used by the
        thing using the multiprocessing based executor) we need to establish
        a link between that process and this process that ensures that when a
        event is triggered in that task in that process that a corresponding
        event is triggered on the original task that was requested to be ran
        in this process.

        To accomplish this we have to create a copy of the task (without
        any listeners) and then reattach a new set of listeners that will
        now instead of calling the desired listeners just place messages
        for this process (a dispatcher thread that is created in this class)
        to dispatch to the original task (using a common accepting socket and
        per task sender socket that is used and associated to know
        which task to proxy back too, since it is possible that there many
        be *many* subprocess running at the same time).

        Once the subprocess task has finished execution, the executor will
        then trigger a callback that will remove the task + target from the
        dispatcher (which will stop any further proxying back to the original
        task).
        