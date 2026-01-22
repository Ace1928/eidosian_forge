import sys
import argparse
import os
import warnings
from . import loader, runner
from .signals import installHandler
def _getDiscoveryArgParser(self, parent):
    parser = argparse.ArgumentParser(parents=[parent])
    parser.prog = '%s discover' % self.progName
    parser.epilog = 'For test discovery all test modules must be importable from the top level directory of the project.'
    parser.add_argument('-s', '--start-directory', dest='start', help="Directory to start discovery ('.' default)")
    parser.add_argument('-p', '--pattern', dest='pattern', help="Pattern to match tests ('test*.py' default)")
    parser.add_argument('-t', '--top-level-directory', dest='top', help='Top level directory of project (defaults to start directory)')
    for arg in ('start', 'pattern', 'top'):
        parser.add_argument(arg, nargs='?', default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    return parser