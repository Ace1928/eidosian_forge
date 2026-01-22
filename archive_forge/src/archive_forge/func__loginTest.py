from __future__ import annotations
from twisted.cred import credentials, error
from twisted.cred.checkers import FilePasswordDB
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words import tap
def _loginTest(self, opt: tap.Options) -> defer.Deferred[None]:
    """
        This method executes both positive and negative authentication
        tests against whatever credentials checker has been stored in
        the Options class.

        @param opt: An instance of L{tap.Options}.
        """
    self.assertEqual(len(opt['credCheckers']), 1)
    checker: FilePasswordDB = opt['credCheckers'][0]
    self.assertFailure(checker.requestAvatarId(self.joeWrong), error.UnauthorizedLogin)

    def _gotAvatar(username: bytes | tuple[()]) -> None:
        self.assertEqual(username, self.admin.username)
    return checker.requestAvatarId(self.admin).addCallback(_gotAvatar)