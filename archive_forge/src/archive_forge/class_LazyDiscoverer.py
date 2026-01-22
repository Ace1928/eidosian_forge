import json
import os
from collections import defaultdict
import hashlib
import tempfile
from functools import partial
import kubernetes.dynamic
import kubernetes.dynamic.discovery
from kubernetes import __version__
from kubernetes.dynamic.exceptions import (
from ansible_collections.kubernetes.core.plugins.module_utils.client.resource import (
class LazyDiscoverer(Discoverer, kubernetes.dynamic.LazyDiscoverer):

    def __init__(self, client, cache_file):
        Discoverer.__init__(self, client, cache_file)
        self.__update_cache = False

    @property
    def update_cache(self):
        self.__update_cache