from __future__ import print_function
import os
import fixtures
from pbr import git
from pbr import options
from pbr.tests import base
def _make_old_git_changelog_format(line):
    """Convert post-1.8.1 git log format to pre-1.8.1 git log format"""
    if not line.strip():
        return line
    sha, msg, refname = line.split('\x00')
    refname = refname.replace('tag: ', '')
    return '\x00'.join((sha, msg, refname))