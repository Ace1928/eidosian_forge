from __future__ import division
from __future__ import print_function
import collections
import contextlib
import errno
import functools
import os
import socket
import stat
import sys
import threading
import warnings
from collections import namedtuple
from socket import AF_INET
from socket import SOCK_DGRAM
from socket import SOCK_STREAM
class _WrapNumbers:
    """Watches numbers so that they don't overflow and wrap
    (reset to zero).
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.cache = {}
        self.reminders = {}
        self.reminder_keys = {}

    def _add_dict(self, input_dict, name):
        assert name not in self.cache
        assert name not in self.reminders
        assert name not in self.reminder_keys
        self.cache[name] = input_dict
        self.reminders[name] = collections.defaultdict(int)
        self.reminder_keys[name] = collections.defaultdict(set)

    def _remove_dead_reminders(self, input_dict, name):
        """In case the number of keys changed between calls (e.g. a
        disk disappears) this removes the entry from self.reminders.
        """
        old_dict = self.cache[name]
        gone_keys = set(old_dict.keys()) - set(input_dict.keys())
        for gone_key in gone_keys:
            for remkey in self.reminder_keys[name][gone_key]:
                del self.reminders[name][remkey]
            del self.reminder_keys[name][gone_key]

    def run(self, input_dict, name):
        """Cache dict and sum numbers which overflow and wrap.
        Return an updated copy of `input_dict`.
        """
        if name not in self.cache:
            self._add_dict(input_dict, name)
            return input_dict
        self._remove_dead_reminders(input_dict, name)
        old_dict = self.cache[name]
        new_dict = {}
        for key in input_dict:
            input_tuple = input_dict[key]
            try:
                old_tuple = old_dict[key]
            except KeyError:
                new_dict[key] = input_tuple
                continue
            bits = []
            for i in range(len(input_tuple)):
                input_value = input_tuple[i]
                old_value = old_tuple[i]
                remkey = (key, i)
                if input_value < old_value:
                    self.reminders[name][remkey] += old_value
                    self.reminder_keys[name][key].add(remkey)
                bits.append(input_value + self.reminders[name][remkey])
            new_dict[key] = tuple(bits)
        self.cache[name] = input_dict
        return new_dict

    def cache_clear(self, name=None):
        """Clear the internal cache, optionally only for function 'name'."""
        with self.lock:
            if name is None:
                self.cache.clear()
                self.reminders.clear()
                self.reminder_keys.clear()
            else:
                self.cache.pop(name, None)
                self.reminders.pop(name, None)
                self.reminder_keys.pop(name, None)

    def cache_info(self):
        """Return internal cache dicts as a tuple of 3 elements."""
        with self.lock:
            return (self.cache, self.reminders, self.reminder_keys)