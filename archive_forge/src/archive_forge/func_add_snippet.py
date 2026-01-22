import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def add_snippet(self, snippet_id, code):
    """A a snippet of code to be processed"""
    self.snippets_id.append(snippet_id)
    self.snippets_code.append(code)