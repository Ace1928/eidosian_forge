from os import chdir, devnull, getcwd
from subprocess import PIPE, Popen
from sys import executable
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.test.test_shellcomp import ZshScriptTestMixin
from twisted.trial.unittest import SkipTest, TestCase

        The trial script adds the current working directory to sys.path so that
        it's able to import modules from it.
        