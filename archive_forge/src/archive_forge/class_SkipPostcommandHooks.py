from typing import (
class SkipPostcommandHooks(Exception):
    """
    Custom exception class for when a command has a failure bad enough to skip post command
    hooks, but not bad enough to print the exception to the user.
    """
    pass