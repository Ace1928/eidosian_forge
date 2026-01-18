import os
import tempfile
import eventlet
from eventlet import greenpool
from oslotest import base as test_base
from oslo_concurrency import lockutils
Verify spawn_n greenthreads with two locks run concurrently.