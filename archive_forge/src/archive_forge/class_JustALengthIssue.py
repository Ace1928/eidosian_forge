from __future__ import with_statement
import textwrap
from difflib import ndiff
from io import open
from os import listdir
from os.path import dirname, isdir, join, realpath, relpath, splitext
import pytest
import chardet
class JustALengthIssue(Exception):
    pass