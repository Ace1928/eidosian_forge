import sys
import argparse
import os
import warnings
from . import loader, runner
from .signals import installHandler
def _do_discovery(self, argv, Loader=None):
    self.start = '.'
    self.pattern = 'test*.py'
    self.top = None
    if argv is not None:
        if self._discovery_parser is None:
            self._initArgParsers()
        self._discovery_parser.parse_args(argv, self)
    self.createTests(from_discovery=True, Loader=Loader)