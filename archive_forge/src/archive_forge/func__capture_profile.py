import errno
import gc
import logging
import os
import pprint
import sys
import tempfile
import traceback
import eventlet.backdoor
import greenlet
import yappi
from eventlet.green import socket
from oslo_service._i18n import _
from oslo_service import _options
def _capture_profile(fname=''):
    if not fname:
        yappi.set_clock_type('cpu')
        yappi.set_context_id_callback(lambda: id(greenlet.getcurrent()))
        yappi.set_context_name_callback(lambda: greenlet.getcurrent().__class__.__name__)
        yappi.start()
    else:
        yappi.stop()
        stats = yappi.get_func_stats()
        try:
            stats_file = os.path.join(tempfile.gettempdir(), fname + '.prof')
            stats.save(stats_file, 'pstat')
        except Exception as e:
            print('Error while saving the trace stats ', str(e))
        finally:
            yappi.clear_stats()