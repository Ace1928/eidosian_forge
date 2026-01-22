import contextlib
import copy
import json
import os
import subprocess
import sys
import threading
import unittest
import six
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.eager import context
from tensorflow.python.eager import remote
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator
from tensorflow.python.training import server_lib
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
class MockOsEnv(collections_abc.Mapping):
    """A class that allows per-thread TF_CONFIG."""

    def __init__(self, *args):
        self._dict = dict()
        self._thread_local = threading.local()
        super(MockOsEnv, self).__init__(*args)

    def get(self, key, default=None):
        if not hasattr(self._thread_local, 'dict'):
            self._thread_local.dict = dict()
        if key == 'TF_CONFIG':
            return dict.get(self._thread_local.dict, key, default)
        else:
            return dict.get(self._dict, key, default)

    def __getitem__(self, key):
        if not hasattr(self._thread_local, 'dict'):
            self._thread_local.dict = dict()
        if key == 'TF_CONFIG':
            return dict.__getitem__(self._thread_local.dict, key)
        else:
            return dict.__getitem__(self._dict, key)

    def __setitem__(self, key, val):
        if not hasattr(self._thread_local, 'dict'):
            self._thread_local.dict = dict()
        if key == 'TF_CONFIG':
            return dict.__setitem__(self._thread_local.dict, key, val)
        else:
            return dict.__setitem__(self._dict, key, val)

    def __iter__(self):
        if not hasattr(self._thread_local, 'dict'):
            self._thread_local.dict = dict()
        for x in self._thread_local.dict:
            yield x
        for x in self._dict:
            yield x

    def __len__(self):
        if not hasattr(self._thread_local, 'dict'):
            self._thread_local.dict = dict()
        return self._thread_local.dict.__len__() + self._dict.__len__()