import code
import greenlet
import logging
import signal
from curtsies.input import is_main_thread
def writelines(self, l):
    for s in l:
        self.write(s)