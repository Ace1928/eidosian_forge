import sys
from io import BytesIO
from typing import List, Optional
from twisted.python import _shellcomp, reflect, usage
from twisted.python.usage import CompleteFiles, CompleteList, Completer, Completions
from twisted.trial import unittest
class ZshTests(unittest.TestCase):
    """
    Tests for zsh completion code
    """

    def test_accumulateMetadata(self):
        """
        Are `compData' attributes you can place on Options classes
        picked up correctly?
        """
        opts = FighterAceExtendedOptions()
        ag = _shellcomp.ZshArgumentsGenerator(opts, 'ace', BytesIO())
        descriptions = FighterAceOptions.compData.descriptions.copy()
        descriptions.update(FighterAceExtendedOptions.compData.descriptions)
        self.assertEqual(ag.descriptions, descriptions)
        self.assertEqual(ag.multiUse, set(FighterAceOptions.compData.multiUse))
        self.assertEqual(ag.mutuallyExclusive, FighterAceOptions.compData.mutuallyExclusive)
        optActions = FighterAceOptions.compData.optActions.copy()
        optActions.update(FighterAceExtendedOptions.compData.optActions)
        self.assertEqual(ag.optActions, optActions)
        self.assertEqual(ag.extraActions, FighterAceOptions.compData.extraActions)

    def test_mutuallyExclusiveCornerCase(self):
        """
        Exercise a corner-case of ZshArgumentsGenerator.makeExcludesDict()
        where the long option name already exists in the `excludes` dict being
        built.
        """

        class OddFighterAceOptions(FighterAceExtendedOptions):
            optFlags = [['anatra', None, 'Select the Anatra DS as your dogfighter aircraft']]
            compData = Completions(mutuallyExclusive=[['anatra', 'fokker', 'albatros', 'spad', 'bristol']])
        opts = OddFighterAceOptions()
        ag = _shellcomp.ZshArgumentsGenerator(opts, 'ace', BytesIO())
        expected = {'albatros': {'anatra', 'b', 'bristol', 'f', 'fokker', 's', 'spad'}, 'anatra': {'a', 'albatros', 'b', 'bristol', 'f', 'fokker', 's', 'spad'}, 'bristol': {'a', 'albatros', 'anatra', 'f', 'fokker', 's', 'spad'}, 'fokker': {'a', 'albatros', 'anatra', 'b', 'bristol', 's', 'spad'}, 'spad': {'a', 'albatros', 'anatra', 'b', 'bristol', 'f', 'fokker'}}
        self.assertEqual(ag.excludes, expected)

    def test_accumulateAdditionalOptions(self):
        """
        We pick up options that are only defined by having an
        appropriately named method on your Options class,
        e.g. def opt_foo(self, foo)
        """
        opts = FighterAceExtendedOptions()
        ag = _shellcomp.ZshArgumentsGenerator(opts, 'ace', BytesIO())
        self.assertIn('nocrash', ag.flagNameToDefinition)
        self.assertIn('nocrash', ag.allOptionsNameToDefinition)
        self.assertIn('difficulty', ag.paramNameToDefinition)
        self.assertIn('difficulty', ag.allOptionsNameToDefinition)

    def test_verifyZshNames(self):
        """
        Using a parameter/flag name that doesn't exist
        will raise an error
        """

        class TmpOptions(FighterAceExtendedOptions):
            compData = Completions(optActions={'detaill': None})
        self.assertRaises(ValueError, _shellcomp.ZshArgumentsGenerator, TmpOptions(), 'ace', BytesIO())

        class TmpOptions2(FighterAceExtendedOptions):
            compData = Completions(mutuallyExclusive=[('foo', 'bar')])
        self.assertRaises(ValueError, _shellcomp.ZshArgumentsGenerator, TmpOptions2(), 'ace', BytesIO())

    def test_zshCode(self):
        """
        Generate a completion function, and test the textual output
        against a known correct output
        """
        outputFile = BytesIO()
        self.patch(usage.Options, '_shellCompFile', outputFile)
        self.patch(sys, 'argv', ['silly', '', '--_shell-completion', 'zsh:2'])
        opts = SimpleProgOptions()
        self.assertRaises(SystemExit, opts.parseOptions)
        self.assertEqual(testOutput1, outputFile.getvalue())

    def test_zshCodeWithSubs(self):
        """
        Generate a completion function with subcommands,
        and test the textual output against a known correct output
        """
        outputFile = BytesIO()
        self.patch(usage.Options, '_shellCompFile', outputFile)
        self.patch(sys, 'argv', ['silly2', '', '--_shell-completion', 'zsh:2'])
        opts = SimpleProgWithSubcommands()
        self.assertRaises(SystemExit, opts.parseOptions)
        self.assertEqual(testOutput2, outputFile.getvalue())

    def test_incompleteCommandLine(self):
        """
        Completion still happens even if a command-line is given
        that would normally throw UsageError.
        """
        outputFile = BytesIO()
        self.patch(usage.Options, '_shellCompFile', outputFile)
        opts = FighterAceOptions()
        self.assertRaises(SystemExit, opts.parseOptions, ['--fokker', 'server', '--unknown-option', '--unknown-option2', '--_shell-completion', 'zsh:5'])
        outputFile.seek(0)
        self.assertEqual(1, len(outputFile.read(1)))

    def test_incompleteCommandLine_case2(self):
        """
        Completion still happens even if a command-line is given
        that would normally throw UsageError.

        The existence of --unknown-option prior to the subcommand
        will break subcommand detection... but we complete anyway
        """
        outputFile = BytesIO()
        self.patch(usage.Options, '_shellCompFile', outputFile)
        opts = FighterAceOptions()
        self.assertRaises(SystemExit, opts.parseOptions, ['--fokker', '--unknown-option', 'server', '--list-server', '--_shell-completion', 'zsh:5'])
        outputFile.seek(0)
        self.assertEqual(1, len(outputFile.read(1)))
        outputFile.seek(0)
        outputFile.truncate()

    def test_incompleteCommandLine_case3(self):
        """
        Completion still happens even if a command-line is given
        that would normally throw UsageError.

        Break subcommand detection in a different way by providing
        an invalid subcommand name.
        """
        outputFile = BytesIO()
        self.patch(usage.Options, '_shellCompFile', outputFile)
        opts = FighterAceOptions()
        self.assertRaises(SystemExit, opts.parseOptions, ['--fokker', 'unknown-subcommand', '--list-server', '--_shell-completion', 'zsh:4'])
        outputFile.seek(0)
        self.assertEqual(1, len(outputFile.read(1)))

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

    def test_poorlyDescribedOptMethod(self):
        """
        Test corner case fetching an option description from a method docstring
        """
        opts = FighterAceOptions()
        argGen = _shellcomp.ZshArgumentsGenerator(opts, 'ace', None)
        descr = argGen.getDescription('silly')
        self.assertEqual(descr, 'silly')

    def test_brokenActions(self):
        """
        A C{Completer} with repeat=True may only be used as the
        last item in the extraActions list.
        """

        class BrokenActions(usage.Options):
            compData = usage.Completions(extraActions=[usage.Completer(repeat=True), usage.Completer()])
        outputFile = BytesIO()
        opts = BrokenActions()
        self.patch(opts, '_shellCompFile', outputFile)
        self.assertRaises(ValueError, opts.parseOptions, ['', '--_shell-completion', 'zsh:2'])

    def test_optMethodsDontOverride(self):
        """
        opt_* methods on Options classes should not override the
        data provided in optFlags or optParameters.
        """

        class Options(usage.Options):
            optFlags = [['flag', 'f', 'A flag']]
            optParameters = [['param', 'p', None, 'A param']]

            def opt_flag(self):
                """junk description"""

            def opt_param(self, param):
                """junk description"""
        opts = Options()
        argGen = _shellcomp.ZshArgumentsGenerator(opts, 'ace', None)
        self.assertEqual(argGen.getDescription('flag'), 'A flag')
        self.assertEqual(argGen.getDescription('param'), 'A param')