from __future__ import print_function
import functools
import os
import subprocess
from unittest import TestCase, skipIf
import attr
from .._methodical import MethodicalMachine
from .test_discover import isTwistedInstalled
def test_saveDotAndImagesInSameDirectory(self):
    """
        Passing the same directory to --image-directory and --dot-directory
        writes images and dot files to that one directory.
        """
    directory = 'imagesAndDot'
    self.tool(argv=[self.fakeFQPN, '--image-directory', directory, '--dot-directory', directory])
    self.assertTrue(any(('image and dot' in line for line in self.collectedOutput)))
    self.assertEqual(len(self.digraphRecorder.renderCalls), 1)
    renderCall, = self.digraphRecorder.renderCalls
    self.assertEqual(renderCall['directory'], directory)
    self.assertFalse(renderCall['cleanup'])
    self.assertFalse(len(self.digraphRecorder.saveCalls))