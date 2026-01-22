import asyncio
import asyncio.events
import collections
import contextlib
import gc
import logging
import os
import pprint
import re
import select
import socket
import ssl
import sys
import tempfile
import threading
import time
import unittest
import uvloop
class BaseTestCaseMeta(type):

    @classmethod
    def __prepare__(mcls, name, bases):
        return TestCaseDict(name)

    def __new__(mcls, name, bases, dct):
        for test_name in dct:
            if not test_name.startswith('test_'):
                continue
            for base in bases:
                if hasattr(base, test_name):
                    raise RuntimeError('duplicate test {}.{} (also defined in {} parent class)'.format(name, test_name, base.__name__))
        return super().__new__(mcls, name, bases, dict(dct))