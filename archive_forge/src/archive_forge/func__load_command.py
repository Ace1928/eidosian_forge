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
def _load_command(self, manager, command_name):
    """Load a command using an instance of a `CommandManager`."""
    try:
        return manager.find_command(command_name.split())[0]
    except ValueError:
        raise self.error('"{}" is not a valid command in the "{}" namespace'.format(command_name, manager.namespace))