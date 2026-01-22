import os
import subprocess
import sys
from .error import TryNext
class CommandChainDispatcher:
    """ Dispatch calls to a chain of commands until some func can handle it

    Usage: instantiate, execute "add" to add commands (with optional
    priority), execute normally via f() calling mechanism.

    """

    def __init__(self, commands=None):
        if commands is None:
            self.chain = []
        else:
            self.chain = commands

    def __call__(self, *args, **kw):
        """ Command chain is called just like normal func.

        This will call all funcs in chain with the same args as were given to
        this function, and return the result of first func that didn't raise
        TryNext"""
        last_exc = TryNext()
        for prio, cmd in self.chain:
            try:
                return cmd(*args, **kw)
            except TryNext as exc:
                last_exc = exc
        raise last_exc

    def __str__(self):
        return str(self.chain)

    def add(self, func, priority=0):
        """ Add a func to the cmd chain with given priority """
        self.chain.append((priority, func))
        self.chain.sort(key=lambda x: x[0])

    def __iter__(self):
        """ Return all objects in chain.

        Handy if the objects are not callable.
        """
        return iter(self.chain)