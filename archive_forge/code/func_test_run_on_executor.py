import asyncio
import contextlib
import gc
import io
import sys
import traceback
import types
import typing
import unittest
import tornado
from tornado import web, gen, httpclient
from tornado.test.util import skipNotCPython
def test_run_on_executor(self):
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(1) as thread_pool:

        class Factory(object):
            executor = thread_pool

            @tornado.concurrent.run_on_executor
            def run(self):
                return None
        factory = Factory()

        async def main():
            for i in range(2):
                await factory.run()
        with assert_no_cycle_garbage():
            asyncio.run(main())