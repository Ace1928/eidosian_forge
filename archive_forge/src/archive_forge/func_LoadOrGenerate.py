from a man-ish style runtime document.
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import json
import os
import re
import shlex
import subprocess
import tarfile
import textwrap
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.command_lib.static_completion import generate as generate_static
from googlecloudsdk.command_lib.static_completion import lookup
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
from six.moves import range
def LoadOrGenerate(self, directories=None, force=False, generate=True, ignore_out_of_date=False, tarball=False, verbose=False, warn_on_exceptions=False):
    """Loads the CLI tree or generates it if necessary, and returns the tree."""
    f = None
    try:
        path, f = self.FindTreeFile(directories)
        if f:
            up_to_date = False
            try:
                tree = json.load(f)
            except ValueError:
                tree = None
            if tree:
                readonly, up_to_date = self.IsUpToDate(tree, verbose=verbose)
                if readonly:
                    return tree
                elif up_to_date:
                    if not force:
                        return tree
                elif ignore_out_of_date:
                    return None
    finally:
        if f:
            f.close()

    def _Generate():
        """Helper that generates a CLI tree and writes it to a JSON file."""
        tree = self.Generate()
        if tree:
            try:
                f = files.FileWriter(path)
            except files.Error as e:
                directory, _ = os.path.split(path)
                try:
                    files.MakeDir(directory)
                    f = files.FileWriter(path)
                except files.Error:
                    if not warn_on_exceptions:
                        raise
                    log.warning(six.text_type(e))
                    return None
            with f:
                resource_printer.Print(tree, print_format='json', out=f)
        return tree
    if not generate:
        raise NoCliTreeForCommandError('No CLI tree for [{}].'.format(self.command_name))
    if not verbose:
        return _Generate()
    with progress_tracker.ProgressTracker('{} the [{}] CLI tree'.format('Updating' if f else 'Generating', self.command_name)):
        return _Generate()