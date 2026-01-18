from os import chdir, devnull, getcwd
from subprocess import PIPE, Popen
from sys import executable
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.test.test_shellcomp import ZshScriptTestMixin
from twisted.trial.unittest import SkipTest, TestCase
def test_twistd(self) -> None:
    self.scriptTest('twistd')