from __future__ import (absolute_import, division, print_function)
import copy
import errno
import os
import tempfile
import time
from abc import abstractmethod
from collections.abc import MutableMapping
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.plugins import AnsiblePlugin
from ansible.plugins.loader import cache_loader
from ansible.utils.collection_loader import resource_from_fqcr
from ansible.utils.display import Display
class BaseCacheModule(AnsiblePlugin):
    _display = display

    def __init__(self, *args, **kwargs):
        super(BaseCacheModule, self).__init__()
        self.set_options(var_options=args, direct=kwargs)

    @abstractmethod
    def get(self, key):
        pass

    @abstractmethod
    def set(self, key, value):
        pass

    @abstractmethod
    def keys(self):
        pass

    @abstractmethod
    def contains(self, key):
        pass

    @abstractmethod
    def delete(self, key):
        pass

    @abstractmethod
    def flush(self):
        pass

    @abstractmethod
    def copy(self):
        pass