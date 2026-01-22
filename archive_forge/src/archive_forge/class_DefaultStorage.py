import warnings
from django.conf import DEFAULT_STORAGE_ALIAS, settings
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.functional import LazyObject
from django.utils.module_loading import import_string
from .base import Storage
from .filesystem import FileSystemStorage
from .handler import InvalidStorageError, StorageHandler
from .memory import InMemoryStorage
class DefaultStorage(LazyObject):

    def _setup(self):
        self._wrapped = storages[DEFAULT_STORAGE_ALIAS]