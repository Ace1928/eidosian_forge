from __future__ import (absolute_import, division, print_function)
import traceback
class connection_object:

    def __init__(self, module):
        self.module = module

    def __enter__(self):
        return setup_conn(self.module)

    def __exit__(self, type, value, traceback):
        close_conn()