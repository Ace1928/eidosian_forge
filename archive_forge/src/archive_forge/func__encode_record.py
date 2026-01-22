import time
import logging
import datetime
import functools
from pyzor.engines.common import *
@staticmethod
def _encode_record(r):
    return {'r_count': r.r_count, 'r_entered': encode_date(r.r_entered), 'r_updated': encode_date(r.r_updated), 'wl_count': r.wl_count, 'wl_entered': encode_date(r.wl_entered), 'wl_updated': encode_date(r.wl_updated)}