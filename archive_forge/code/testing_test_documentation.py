from tornado import gen, ioloop
from tornado.httpserver import HTTPServer
from tornado.locks import Event
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, bind_unused_port, gen_test
from tornado.web import Application
import asyncio
import contextlib
import inspect
import gc
import os
import platform
import sys
import traceback
import unittest
import warnings

        This test makes sure that AsyncTestCase calls super methods for
        setUp and tearDown.

        InheritBoth is a subclass of both AsyncTestCase and
        SetUpTearDown, with the ordering so that the super of
        AsyncTestCase will be SetUpTearDown.
        