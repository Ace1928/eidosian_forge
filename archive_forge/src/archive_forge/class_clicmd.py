import threading, inspect, shlex
class clicmd(object):

    def __init__(self, desc, order=0):
        self.desc = desc
        self.order = order

    def __call__(self, fn):
        fn.clidesc = self.desc
        fn.cliorder = self.order
        return fn