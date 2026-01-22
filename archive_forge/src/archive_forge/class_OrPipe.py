import sys
import os
import socket
class OrPipe:

    def __init__(self, pipe):
        self._set = False
        self._partner = None
        self._pipe = pipe

    def set(self):
        self._set = True
        if not self._partner._set:
            self._pipe.set()

    def clear(self):
        self._set = False
        if not self._partner._set:
            self._pipe.clear()