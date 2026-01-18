from __future__ import annotations
import argparse
import os
import sys
from argparse import ArgumentTypeError
def version_f(args):
    from .. import __version__
    print('wheel %s' % __version__)