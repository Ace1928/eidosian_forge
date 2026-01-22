from __future__ import print_function
import functools
import os
import subprocess
from unittest import TestCase, skipIf
import attr
from .._methodical import MethodicalMachine
from .test_discover import isTwistedInstalled
@skipIf(not isGraphvizModuleInstalled(), 'Graphviz module is not installed.')
@skipIf(not isGraphvizInstalled(), 'Graphviz tools are not installed.')
@skipIf(not isTwistedInstalled(), 'Twisted is not installed.')
class VisualizeToolTests(TestCase):

    def setUp(self):
        self.digraphRecorder = RecordsDigraphActions()
        self.fakeDigraph = FakeDigraph(self.digraphRecorder)
        self.fakeProgname = 'tool-test'
        self.fakeSysPath = ['ignored']
        self.collectedOutput = []
        self.fakeFQPN = 'fake.fqpn'

    def collectPrints(self, *args):
        self.collectedOutput.append(' '.join(args))

    def fakeFindMachines(self, fqpn):
        yield (fqpn, FakeMethodicalMachine(self.fakeDigraph))

    def tool(self, progname=None, argv=None, syspath=None, findMachines=None, print=None):
        from .._visualize import tool
        return tool(_progname=progname or self.fakeProgname, _argv=argv or [self.fakeFQPN], _syspath=syspath or self.fakeSysPath, _findMachines=findMachines or self.fakeFindMachines, _print=print or self.collectPrints)

    def test_checksCurrentDirectory(self):
        """
        L{tool} adds '' to sys.path to ensure
        L{automat._discover.findMachines} searches the current
        directory.
        """
        self.tool(argv=[self.fakeFQPN])
        self.assertEqual(self.fakeSysPath[0], '')

    def test_quietHidesOutput(self):
        """
        Passing -q/--quiet hides all output.
        """
        self.tool(argv=[self.fakeFQPN, '--quiet'])
        self.assertFalse(self.collectedOutput)
        self.tool(argv=[self.fakeFQPN, '-q'])
        self.assertFalse(self.collectedOutput)

    def test_onlySaveDot(self):
        """
        Passing an empty string for --image-directory/-i disables
        rendering images.
        """
        for arg in ('--image-directory', '-i'):
            self.digraphRecorder.reset()
            self.collectedOutput = []
            self.tool(argv=[self.fakeFQPN, arg, ''])
            self.assertFalse(any(('image' in line for line in self.collectedOutput)))
            self.assertEqual(len(self.digraphRecorder.saveCalls), 1)
            call, = self.digraphRecorder.saveCalls
            self.assertEqual('{}.dot'.format(self.fakeFQPN), call['filename'])
            self.assertFalse(self.digraphRecorder.renderCalls)

    def test_saveOnlyImage(self):
        """
        Passing an empty string for --dot-directory/-d disables saving dot
        files.
        """
        for arg in ('--dot-directory', '-d'):
            self.digraphRecorder.reset()
            self.collectedOutput = []
            self.tool(argv=[self.fakeFQPN, arg, ''])
            self.assertFalse(any(('dot' in line for line in self.collectedOutput)))
            self.assertEqual(len(self.digraphRecorder.renderCalls), 1)
            call, = self.digraphRecorder.renderCalls
            self.assertEqual('{}.dot'.format(self.fakeFQPN), call['filename'])
            self.assertTrue(call['cleanup'])
            self.assertFalse(self.digraphRecorder.saveCalls)

    def test_saveDotAndImagesInDifferentDirectories(self):
        """
        Passing different directories to --image-directory and --dot-directory
        writes images and dot files to those directories.
        """
        imageDirectory = 'image'
        dotDirectory = 'dot'
        self.tool(argv=[self.fakeFQPN, '--image-directory', imageDirectory, '--dot-directory', dotDirectory])
        self.assertTrue(any(('image' in line for line in self.collectedOutput)))
        self.assertTrue(any(('dot' in line for line in self.collectedOutput)))
        self.assertEqual(len(self.digraphRecorder.renderCalls), 1)
        renderCall, = self.digraphRecorder.renderCalls
        self.assertEqual(renderCall['directory'], imageDirectory)
        self.assertTrue(renderCall['cleanup'])
        self.assertEqual(len(self.digraphRecorder.saveCalls), 1)
        saveCall, = self.digraphRecorder.saveCalls
        self.assertEqual(saveCall['directory'], dotDirectory)

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