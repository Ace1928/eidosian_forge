import argparse
import optparse
import os
import signal
import sys
from getopt import getopt
from typing import ClassVar, Dict, Optional, Type
from dulwich import porcelain
from .client import GitProtocolError, get_transport_and_path
from .errors import ApplyDeltaError
from .index import Index
from .objectspec import parse_commit
from .pack import Pack, sha_to_hex
from .repo import Repo
class cmd_diff(Command):

    def run(self, args):
        opts, args = getopt(args, '', [])
        r = Repo('.')
        if args == []:
            commit_id = b'HEAD'
        else:
            commit_id = args[0]
        commit = parse_commit(r, commit_id)
        parent_commit = r[commit.parents[0]]
        porcelain.diff_tree(r, parent_commit.tree, commit.tree, outstream=sys.stdout.buffer)