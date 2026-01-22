import itertools
import shlex
import sys
import autopage.argparse
import cmd2
def _complete_prefix(self, prefix):
    """Returns cliff style commands with a specific prefix."""
    if not prefix:
        return [n for n, v in self.command_manager]
    return [n for n, v in self.command_manager if n.startswith(prefix)]