from __future__ import print_function
import os
import fixtures
from pbr import git
from pbr import options
from pbr.tests import base
def test_write_git_changelog(self):
    self.useFixture(fixtures.FakePopen(lambda _: {'stdout': BytesIO(self.changelog.encode('utf-8'))}))
    git.write_git_changelog(git_dir=self.git_dir, dest_dir=self.temp_path)
    with open(os.path.join(self.temp_path, 'ChangeLog'), 'r') as ch_fh:
        changelog_contents = ch_fh.read()
        self.assertIn('2013.2', changelog_contents)
        self.assertIn('0.5.17', changelog_contents)
        self.assertIn('------', changelog_contents)
        self.assertIn('Refactor hooks file', changelog_contents)
        self.assertIn('Bug fix: create\\_stack() fails when waiting', changelog_contents)
        self.assertNotIn('Refactor hooks file.', changelog_contents)
        self.assertNotIn('182feb3', changelog_contents)
        self.assertNotIn('review/monty_taylor/27519', changelog_contents)
        self.assertNotIn('0.5.13', changelog_contents)
        self.assertNotIn('0.6.7', changelog_contents)
        self.assertNotIn('12', changelog_contents)
        self.assertNotIn('(evil)', changelog_contents)
        self.assertNotIn('ev()il', changelog_contents)
        self.assertNotIn('ev(il', changelog_contents)
        self.assertNotIn('ev)il', changelog_contents)
        self.assertNotIn('e(vi)l', changelog_contents)
        self.assertNotIn('Merge "', changelog_contents)
        self.assertNotIn('1\\_foo.1', changelog_contents)