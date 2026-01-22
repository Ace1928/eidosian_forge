import time
import logging
import datetime
import functools
from pyzor.engines.common import *
def _iteritems(self):
    for key in self:
        try:
            yield (key, self[key])
        except Exception as ex:
            self.log.warning('Invalid record %s: %s', key, ex)