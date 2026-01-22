from pickle import PicklingError
import re
import os
import os.path
import datetime
import json
import shutil
import warnings
import collections
import operator
import threading
from abc import ABCMeta, abstractmethod
from .backports import concurrency_safe_rename
from .disk import mkdirp, memstr_to_bytes, rm_subdirs
from . import numpy_pickle
def _get_items_to_delete(self, bytes_limit, items_limit=None, age_limit=None):
    """
        Get items to delete to keep the store under size, file, & age limits.
        """
    if isinstance(bytes_limit, str):
        bytes_limit = memstr_to_bytes(bytes_limit)
    items = self.get_items()
    size = sum((item.size for item in items))
    if bytes_limit is not None:
        to_delete_size = size - bytes_limit
    else:
        to_delete_size = 0
    if items_limit is not None:
        to_delete_items = len(items) - items_limit
    else:
        to_delete_items = 0
    if age_limit is not None:
        older_item = min((item.last_access for item in items))
        deadline = datetime.datetime.now() - age_limit
    else:
        deadline = None
    if to_delete_size <= 0 and to_delete_items <= 0 and (deadline is None or older_item > deadline):
        return []
    items.sort(key=operator.attrgetter('last_access'))
    items_to_delete = []
    size_so_far = 0
    items_so_far = 0
    for item in items:
        if size_so_far >= to_delete_size and items_so_far >= to_delete_items and (deadline is None or deadline < item.last_access):
            break
        items_to_delete.append(item)
        size_so_far += item.size
        items_so_far += 1
    return items_to_delete