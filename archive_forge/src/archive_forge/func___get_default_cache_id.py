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
def __get_default_cache_id(self):
    user = self.__get_user()
    if user:
        cache_id = '{0}-{1}'.format(self.client.configuration.host, user)
    else:
        cache_id = self.client.configuration.host
    return cache_id.encode('utf-8')