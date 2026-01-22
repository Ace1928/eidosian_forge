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
def _print_greenthreads(simple=True):
    for i, gt in enumerate(_find_objects(greenlet.greenlet)):
        print(i, gt)
        if simple:
            traceback.print_stack(gt.gr_frame)
        else:
            _detailed_dump_frames(gt.gr_frame, i)
        print()