import itertools
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
import weakref
from collections import defaultdict
from functools import lru_cache, wraps
from pathlib import Path
from types import ModuleType
from zipimport import zipimporter
import django
from django.apps import apps
from django.core.signals import request_finished
from django.dispatch import Signal
from django.utils.functional import cached_property
from django.utils.version import get_version_tuple
def _watch_root(self, root):
    if not root.exists():
        if not root.parent.exists():
            logger.warning('Unable to watch root dir %s as neither it or its parent exist.', root)
            return
        root = root.parent
    result = self.client.query('watch-project', str(root.absolute()))
    if 'warning' in result:
        logger.warning('Watchman warning: %s', result['warning'])
    logger.debug('Watchman watch-project result: %s', result)
    return (result['watch'], result.get('relative_path'))