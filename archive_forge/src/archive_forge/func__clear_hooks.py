import atexit
import codecs
import contextlib
import copy
import difflib
import doctest
import errno
import functools
import itertools
import logging
import math
import os
import platform
import pprint
import random
import re
import shlex
import site
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import unittest
import warnings
from io import BytesIO, StringIO, TextIOWrapper
from typing import Callable, Set
import testtools
from testtools import content
import breezy
from breezy.bzr import chk_map
from .. import branchbuilder
from .. import commands as _mod_commands
from .. import config, controldir, debug, errors, hooks, i18n
from .. import lock as _mod_lock
from .. import lockdir, osutils
from .. import plugin as _mod_plugin
from .. import pyutils, registry, symbol_versioning, trace
from .. import transport as _mod_transport
from .. import ui, urlutils, workingtree
from ..bzr.smart import client, request
from ..tests import TestUtil, fixtures, test_server, treeshape, ui_testing
from ..transport import memory, pathfilter
from testtools.testcase import TestSkipped
def _clear_hooks(self):
    known_hooks = hooks.known_hooks
    self._preserved_hooks = {}
    for key, (parent, name) in known_hooks.iter_parent_objects():
        current_hooks = getattr(parent, name)
        self._preserved_hooks[parent] = (name, current_hooks)
    self._preserved_lazy_hooks = hooks._lazy_hooks
    hooks._lazy_hooks = {}
    self.addCleanup(self._restoreHooks)
    for key, (parent, name) in known_hooks.iter_parent_objects():
        factory = known_hooks.get(key)
        setattr(parent, name, factory())
    request._install_hook()