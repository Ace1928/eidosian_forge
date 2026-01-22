from os import chdir, devnull, getcwd
from subprocess import PIPE, Popen
from sys import executable
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.test.test_shellcomp import ZshScriptTestMixin
from twisted.trial.unittest import SkipTest, TestCase
class ZshIntegrationTests(TestCase, ZshScriptTestMixin):
    """
    Test that zsh completion functions are generated without error
    """
    generateFor = [('twistd', 'twisted.scripts.twistd.ServerOptions'), ('trial', 'twisted.scripts.trial.Options'), ('pyhtmlizer', 'twisted.scripts.htmlizer.Options')]