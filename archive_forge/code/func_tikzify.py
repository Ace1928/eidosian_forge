import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def tikzify(s):
    if s.strip():
        return mreplace(s, '\\,:.()', '-+_*{}')
    else:
        return 'd2tnn%i' % (len(s) + 1)