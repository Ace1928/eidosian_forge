from twisted.python import usage
from twisted.trial import unittest
def test_userAtHost(self):
    """
        CompleteUserAtHost produces zsh shell-code that completes hostnames or
        a word of the form <username>@<hostname>.
        """
    c = usage.CompleteUserAtHost()
    out = c._shellCode('some-option', usage._ZSH)
    self.assertTrue(out.startswith(':host | user@host:'))
    c = usage.CompleteUserAtHost(descr='some action', repeat=True)
    out = c._shellCode('some-option', usage._ZSH)
    self.assertTrue(out.startswith('*:some action:'))