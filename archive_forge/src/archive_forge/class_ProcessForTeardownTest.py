import asyncio
import json
import os
import sys
from concurrent.futures import CancelledError
from multiprocessing import Process
import pytest
from pytest import mark
import zmq
import zmq.asyncio as zaio
class ProcessForTeardownTest(Process):

    def run(self):
        """Leave context, socket and event loop upon implicit disposal"""
        actx = zaio.Context.instance()
        socket = actx.socket(zmq.PAIR)
        socket.bind_to_random_port('tcp://127.0.0.1')

        async def never_ending_task(socket):
            await socket.recv()
        loop = asyncio.new_event_loop()
        coro = asyncio.wait_for(never_ending_task(socket), timeout=1)
        try:
            loop.run_until_complete(coro)
        except asyncio.TimeoutError:
            pass
        else:
            assert False, 'never_ending_task was completed unexpectedly'
        finally:
            loop.close()