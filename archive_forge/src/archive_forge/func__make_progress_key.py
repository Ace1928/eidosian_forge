from abc import ABC
import inspect
import hashlib
@staticmethod
def _make_progress_key(key):
    return key + '-progress'