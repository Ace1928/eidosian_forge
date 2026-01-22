import collections
import gc
import io
import ipaddress
import json
import platform
import socket
import sys
import traceback
from debtcollector import removals
import jinja2
from oslo_utils import reflection
from oslo_utils import strutils
from oslo_utils import timeutils
import stevedore
import webob.dec
import webob.exc
import webob.response
from oslo_middleware import base
from oslo_middleware.healthcheck import opts
@staticmethod
def _get_greenstacks():
    greenstacks = []
    if greenlet is not None:
        buf = io.StringIO()
        for gt in _find_objects(greenlet.greenlet):
            traceback.print_stack(gt.gr_frame, file=buf)
            greenstacks.append(buf.getvalue())
            buf.seek(0)
            buf.truncate()
    return greenstacks