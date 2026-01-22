import os
import platform
import shelve
import sys
import weakref
import zlib
from collections import Counter
from kombu.serialization import pickle, pickle_protocol
from kombu.utils.objects import cached_property
from celery import __version__
from celery.exceptions import WorkerShutdown, WorkerTerminate
from celery.utils.collections import LimitedSet
def _merge_revoked(self, d):
    try:
        self._merge_revoked_v3(d['zrevoked'])
    except KeyError:
        try:
            self._merge_revoked_v2(d.pop('revoked'))
        except KeyError:
            pass
    self._revoked_tasks.purge()