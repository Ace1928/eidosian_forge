import asyncio
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
from collections.abc import Generator
import contextlib
import datetime
import functools
import socket
import subprocess
import sys
import threading
import time
import types
from unittest import mock
import unittest
from tornado.escape import native_str
from tornado import gen
from tornado.ioloop import IOLoop, TimeoutError, PeriodicCallback
from tornado.log import app_log
from tornado.testing import (
from tornado.test.util import (
from tornado.concurrent import Future
import typing
Simulate a series of calls to the PeriodicCallback.

        Pass a list of call durations in seconds (negative values
        work to simulate clock adjustments during the call, or more or
        less equivalently, between calls). This method returns the
        times at which each call would be made.
        