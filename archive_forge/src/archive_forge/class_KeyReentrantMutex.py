import abc
import collections
import re
import threading
from typing import MutableMapping
from typing import MutableSet
import stevedore
class KeyReentrantMutex:

    def __init__(self, key: str, mutex: Mutex, keys: MutableMapping[int, MutableSet[str]]):
        self.key = key
        self.mutex = mutex
        self.keys = keys

    @classmethod
    def factory(cls, mutex):
        keystore: MutableMapping[int, MutableSet[str]] = collections.defaultdict(set)

        def fac(key):
            return KeyReentrantMutex(key, mutex, keystore)
        return fac

    def acquire(self, wait=True):
        current_thread = threading.get_ident()
        keys = self.keys.get(current_thread)
        if keys is not None and self.key not in keys:
            keys.add(self.key)
            return True
        elif self.mutex.acquire(wait=wait):
            self.keys[current_thread].add(self.key)
            return True
        else:
            return False

    def release(self):
        current_thread = threading.get_ident()
        keys = self.keys.get(current_thread)
        assert keys is not None, "this thread didn't do the acquire"
        assert self.key in keys, "No acquire held for key '%s'" % self.key
        keys.remove(self.key)
        if not keys:
            del self.keys[current_thread]
            self.mutex.release()

    def locked(self):
        current_thread = threading.get_ident()
        keys = self.keys.get(current_thread)
        if keys is None:
            return False
        return self.key in keys