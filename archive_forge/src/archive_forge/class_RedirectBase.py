import itertools
import logging
import os
import queue
import re
import signal
import struct
import sys
import threading
import time
from collections import defaultdict
import wandb
class RedirectBase:

    def __init__(self, src, cbs=()):
        """# Arguments.

        `src`: Source stream to be redirected. "stdout" or "stderr".
        `cbs`: tuple/list of callbacks. Each callback should take exactly 1 argument (bytes).

        """
        assert hasattr(sys, src)
        self.src = src
        self.cbs = cbs

    @property
    def src_stream(self):
        return getattr(sys, '__%s__' % self.src)

    @property
    def src_fd(self):
        return self.src_stream.fileno()

    @property
    def src_wrapped_stream(self):
        return getattr(sys, self.src)

    def save(self):
        pass

    def install(self):
        curr_redirect = _redirects.get(self.src)
        if curr_redirect and curr_redirect != self:
            curr_redirect.uninstall()
        _redirects[self.src] = self

    def uninstall(self):
        if _redirects[self.src] != self:
            return
        _redirects[self.src] = None