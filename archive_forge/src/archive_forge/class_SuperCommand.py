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
class SuperCommand(Command):
    subcommands: ClassVar[Dict[str, Type[Command]]] = {}
    default_command: ClassVar[Optional[Type[Command]]] = None

    def run(self, args):
        if not args and (not self.default_command):
            print('Supported subcommands: %s' % ', '.join(self.subcommands.keys()))
            return False
        cmd = args[0]
        try:
            cmd_kls = self.subcommands[cmd]
        except KeyError:
            print('No such subcommand: %s' % args[0])
            return False
        return cmd_kls().run(args[1:])