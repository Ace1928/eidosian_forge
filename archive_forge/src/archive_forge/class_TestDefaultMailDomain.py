import os
import sys
from .. import bedding, osutils, tests
class TestDefaultMailDomain(tests.TestCaseInTempDir):
    """Test retrieving default domain from mailname file"""

    def test_default_mail_domain_simple(self):
        with open('simple', 'w') as f:
            f.write('domainname.com\n')
        r = bedding._get_default_mail_domain('simple')
        self.assertEqual('domainname.com', r)

    def test_default_mail_domain_no_eol(self):
        with open('no_eol', 'w') as f:
            f.write('domainname.com')
        r = bedding._get_default_mail_domain('no_eol')
        self.assertEqual('domainname.com', r)

    def test_default_mail_domain_multiple_lines(self):
        with open('multiple_lines', 'w') as f:
            f.write('domainname.com\nsome other text\n')
        r = bedding._get_default_mail_domain('multiple_lines')
        self.assertEqual('domainname.com', r)