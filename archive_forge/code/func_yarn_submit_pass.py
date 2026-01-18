from __future__ import absolute_import
import os
import sys
import subprocess
import warnings
import logging
import platform
from threading import Thread
from . import opts
from . import tracker
from .util import py_str
def yarn_submit_pass(nworker, nserver, pass_env):
    submit_thread.append(yarn_submit(args, nworker, nserver, pass_env))