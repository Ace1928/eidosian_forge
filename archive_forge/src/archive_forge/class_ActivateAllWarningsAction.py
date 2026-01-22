from __future__ import absolute_import
import sys
import os
from argparse import ArgumentParser, Action, SUPPRESS
from . import Options
class ActivateAllWarningsAction(Action):

    def __call__(self, parser, namespace, values, option_string=None):
        directives = getattr(namespace, 'compiler_directives', {})
        directives.update(Options.extra_warnings)
        namespace.compiler_directives = directives