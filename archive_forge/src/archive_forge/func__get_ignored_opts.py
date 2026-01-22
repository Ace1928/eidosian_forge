import argparse
import fnmatch
import importlib
import inspect
import re
import sys
from docutils import nodes
from docutils.parsers import rst
from docutils.parsers.rst import directives
from docutils import statemachine
from cliff import app
from cliff import commandmanager
def _get_ignored_opts(self):
    global_ignored = self.env.config.autoprogram_cliff_ignored
    local_ignored = self.options.get('ignored', '')
    local_ignored = [x.strip() for x in local_ignored.split(',') if x.strip()]
    return list(set(global_ignored + local_ignored))