import multiprocessing
import requests
from . import thread
from .._compat import queue
def _new_session(self):
    return self._auth(self._initializer(self._session()))