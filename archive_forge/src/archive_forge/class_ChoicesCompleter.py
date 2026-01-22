from __future__ import absolute_import, division, print_function, unicode_literals
import os
import subprocess
from .compat import str, sys_encoding
class ChoicesCompleter(object):

    def __init__(self, choices):
        self.choices = choices

    def _convert(self, choice):
        if isinstance(choice, bytes):
            choice = choice.decode(sys_encoding)
        if not isinstance(choice, str):
            choice = str(choice)
        return choice

    def __call__(self, **kwargs):
        return (self._convert(c) for c in self.choices)