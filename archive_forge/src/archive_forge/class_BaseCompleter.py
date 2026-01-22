import argparse
import os
import subprocess
class BaseCompleter:
    """
    This is the base class that all argcomplete completers should subclass.
    """

    def __call__(self, *, prefix: str, action: argparse.Action, parser: argparse.ArgumentParser, parsed_args: argparse.Namespace):
        raise NotImplementedError('This method should be implemented by a subclass.')