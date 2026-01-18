import ast
import cmd
import signal
import socket
import sys
import termios
from os_ken import cfg
from os_ken.lib import rpc
def try_to_connect(self, verbose=False):
    if self.client:
        return
    try:
        self.connect()
        assert self.client
    except Exception as e:
        if verbose:
            print('connection failure %s' % e)
        raise EOFError