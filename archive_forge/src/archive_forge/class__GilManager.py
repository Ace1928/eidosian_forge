from llvmlite.ir import Constant, IRBuilder
import llvmlite.ir
from numba.core import types, config, cgutils
class _GilManager(object):
    """
    A utility class to handle releasing the GIL and then re-acquiring it
    again.
    """

    def __init__(self, builder, api, argman):
        self.builder = builder
        self.api = api
        self.argman = argman
        self.thread_state = api.save_thread()

    def emit_cleanup(self):
        self.api.restore_thread(self.thread_state)
        self.argman.emit_cleanup()