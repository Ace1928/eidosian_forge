import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
class TestGnuChangelogFormatter(TestCaseForLogFormatter):

    def test_gnu_changelog(self):
        wt = self.make_standard_commit('nicky', authors=[])
        self.assertFormatterResult(b'2005-11-22  Lorem Ipsum  <test@example.com>\n\n\tadd a\n\n', wt.branch, log.GnuChangelogLogFormatter)

    def test_with_authors(self):
        wt = self.make_standard_commit('nicky', authors=['Fooa Fooz <foo@example.com>', 'Bari Baro <bar@example.com>'])
        self.assertFormatterResult(b'2005-11-22  Fooa Fooz  <foo@example.com>\n\n\tadd a\n\n', wt.branch, log.GnuChangelogLogFormatter)

    def test_verbose(self):
        wt = self.make_standard_commit('nicky')
        self.assertFormatterResult(b'2005-11-22  John Doe  <jdoe@example.com>\n\n\t* a:\n\n\tadd a\n\n', wt.branch, log.GnuChangelogLogFormatter, show_log_kwargs=dict(verbose=True))