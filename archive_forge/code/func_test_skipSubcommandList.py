import sys
from io import BytesIO
from typing import List, Optional
from twisted.python import _shellcomp, reflect, usage
from twisted.python.usage import CompleteFiles, CompleteList, Completer, Completions
from twisted.trial import unittest
def test_skipSubcommandList(self):
    """
        Ensure the optimization which skips building the subcommand list
        under certain conditions isn't broken.
        """
    outputFile = BytesIO()
    self.patch(usage.Options, '_shellCompFile', outputFile)
    opts = FighterAceOptions()
    self.assertRaises(SystemExit, opts.parseOptions, ['--alba', '--_shell-completion', 'zsh:2'])
    outputFile.seek(0)
    self.assertEqual(1, len(outputFile.read(1)))