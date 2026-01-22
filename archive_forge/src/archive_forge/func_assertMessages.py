from breezy import osutils, tests
def assertMessages(self, out, must_have=(), must_not_have=()):
    """Check if commit messages are in or not in the output"""
    for m in must_have:
        self.assertContainsRe(out, '\\nmessage:\\n  %s\\n' % m)
    for m in must_not_have:
        self.assertNotContainsRe(out, '\\nmessage:\\n  %s\\n' % m)