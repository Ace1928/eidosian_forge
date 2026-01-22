import ast
import asyncio
import code
import concurrent.futures
import inspect
import sys
import threading
import types
import warnings
from . import futures
class REPLThread(threading.Thread):

    def run(self):
        try:
            banner = f'asyncio REPL {sys.version} on {sys.platform}\nUse "await" directly instead of "asyncio.run()".\nType "help", "copyright", "credits" or "license" for more information.\n{getattr(sys, 'ps1', '>>> ')}import asyncio'
            console.interact(banner=banner, exitmsg='exiting asyncio REPL...')
        finally:
            warnings.filterwarnings('ignore', message='^coroutine .* was never awaited$', category=RuntimeWarning)
            loop.call_soon_threadsafe(loop.stop)