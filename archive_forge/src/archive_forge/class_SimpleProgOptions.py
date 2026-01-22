import sys
from io import BytesIO
from typing import List, Optional
from twisted.python import _shellcomp, reflect, usage
from twisted.python.usage import CompleteFiles, CompleteList, Completer, Completions
from twisted.trial import unittest
class SimpleProgOptions(usage.Options):
    """
    Command-line options for a `Silly` imaginary program
    """
    optFlags = [['color', 'c', 'Turn on color output'], ['gray', 'g', 'Turn on gray-scale output'], ['verbose', 'v', 'Verbose logging (may be specified more than once)']]
    optParameters = [['optimization', None, '5', 'Select the level of optimization (1-5)'], ['accuracy', 'a', '3', 'Select the level of accuracy (1-3)']]
    compData = Completions(descriptions={'color': 'Color on', 'optimization': 'Optimization level'}, multiUse=['verbose'], mutuallyExclusive=[['color', 'gray']], optActions={'optimization': CompleteList(['1', '2', '3', '4', '5'], descr='Optimization?'), 'accuracy': _accuracyAction}, extraActions=[CompleteFiles(descr='output file')])

    def opt_X(self):
        """
        usage.Options does not recognize single-letter opt_ methods
        """