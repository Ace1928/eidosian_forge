import argparse
from ...utils.dataclasses import (
from ..menu import BulletMenu
class SubcommandHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """
    A custom formatter that will remove the usage line from the help message for subcommands.
    """

    def _format_usage(self, usage, actions, groups, prefix):
        usage = super()._format_usage(usage, actions, groups, prefix)
        usage = usage.replace('<command> [<args>] ', '')
        return usage