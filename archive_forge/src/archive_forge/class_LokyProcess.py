import sys
from multiprocessing.context import assert_spawning
from multiprocessing.process import BaseProcess
class LokyProcess(BaseProcess):
    _start_method = 'loky'

    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, daemon=None, init_main_module=False, env=None):
        super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)
        self.env = {} if env is None else env
        self.authkey = self.authkey
        self.init_main_module = init_main_module

    @staticmethod
    def _Popen(process_obj):
        if sys.platform == 'win32':
            from .popen_loky_win32 import Popen
        else:
            from .popen_loky_posix import Popen
        return Popen(process_obj)