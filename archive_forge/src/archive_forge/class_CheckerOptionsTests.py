import os
from io import StringIO
from typing import Sequence, Type
from unittest import skipIf
from zope.interface import Interface
from twisted import plugin
from twisted.cred import checkers, credentials, error, strcred
from twisted.plugins import cred_anonymous, cred_file, cred_unix
from twisted.python import usage
from twisted.python.fakepwd import UserDatabase
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
class CheckerOptionsTests(TestCase):

    def test_createsList(self):
        """
        The C{--auth} command line creates a list in the
        Options instance and appends values to it.
        """
        options = DummyOptions()
        options.parseOptions(['--auth', 'memory'])
        self.assertEqual(len(options['credCheckers']), 1)
        options = DummyOptions()
        options.parseOptions(['--auth', 'memory', '--auth', 'memory'])
        self.assertEqual(len(options['credCheckers']), 2)

    def test_invalidAuthError(self):
        """
        The C{--auth} command line raises an exception when it
        gets a parameter it doesn't understand.
        """
        options = DummyOptions()
        invalidParameter = getInvalidAuthType()
        self.assertRaises(usage.UsageError, options.parseOptions, ['--auth', invalidParameter])
        self.assertRaises(usage.UsageError, options.parseOptions, ['--help-auth-type', invalidParameter])

    def test_createsDictionary(self):
        """
        The C{--auth} command line creates a dictionary mapping supported
        interfaces to the list of credentials checkers that support it.
        """
        options = DummyOptions()
        options.parseOptions(['--auth', 'memory', '--auth', 'anonymous'])
        chd = options['credInterfaces']
        self.assertEqual(len(chd[credentials.IAnonymous]), 1)
        self.assertEqual(len(chd[credentials.IUsernamePassword]), 1)
        chdAnonymous = chd[credentials.IAnonymous][0]
        chdUserPass = chd[credentials.IUsernamePassword][0]
        self.assertTrue(checkers.ICredentialsChecker.providedBy(chdAnonymous))
        self.assertTrue(checkers.ICredentialsChecker.providedBy(chdUserPass))
        self.assertIn(credentials.IAnonymous, chdAnonymous.credentialInterfaces)
        self.assertIn(credentials.IUsernamePassword, chdUserPass.credentialInterfaces)

    def test_credInterfacesProvidesLists(self):
        """
        When two C{--auth} arguments are passed along which support the same
        interface, a list with both is created.
        """
        options = DummyOptions()
        options.parseOptions(['--auth', 'memory', '--auth', 'unix'])
        self.assertEqual(options['credCheckers'], options['credInterfaces'][credentials.IUsernamePassword])

    def test_listDoesNotDisplayDuplicates(self):
        """
        The list for C{--help-auth} does not duplicate items.
        """
        authTypes = []
        options = DummyOptions()
        for cf in options._checkerFactoriesForOptHelpAuth():
            self.assertNotIn(cf.authType, authTypes)
            authTypes.append(cf.authType)

    def test_displaysListCorrectly(self):
        """
        The C{--help-auth} argument correctly displays all
        available authentication plugins, then exits.
        """
        newStdout = StringIO()
        options = DummyOptions()
        options.authOutput = newStdout
        self.assertRaises(SystemExit, options.parseOptions, ['--help-auth'])
        for checkerFactory in strcred.findCheckerFactories():
            self.assertIn(checkerFactory.authType, newStdout.getvalue())

    def test_displaysHelpCorrectly(self):
        """
        The C{--help-auth-for} argument will correctly display the help file
        for a particular authentication plugin.
        """
        newStdout = StringIO()
        options = DummyOptions()
        options.authOutput = newStdout
        self.assertRaises(SystemExit, options.parseOptions, ['--help-auth-type', 'file'])
        for line in cred_file.theFileCheckerFactory.authHelp:
            if line.strip():
                self.assertIn(line.strip(), newStdout.getvalue())

    def test_unexpectedException(self):
        """
        When the checker specified by C{--auth} raises an unexpected error, it
        should be caught and re-raised within a L{usage.UsageError}.
        """
        options = DummyOptions()
        err = self.assertRaises(usage.UsageError, options.parseOptions, ['--auth', 'file'])
        self.assertEqual(str(err), "Unexpected error: 'file' requires a filename")