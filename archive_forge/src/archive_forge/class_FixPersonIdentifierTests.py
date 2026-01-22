from dulwich.objects import Blob, Commit, Tag, parse_timezone
from dulwich.tests.utils import make_object
from ...revision import Revision
from .. import tests
from ..mapping import (BzrGitMappingv1, UnknownCommitEncoding,
class FixPersonIdentifierTests(tests.TestCase):

    def test_valid(self):
        self.assertEqual(b'foo <bar@blah.nl>', fix_person_identifier(b'foo <bar@blah.nl>'))
        self.assertEqual(b'bar@blah.nl <bar@blah.nl>', fix_person_identifier(b'bar@blah.nl'))

    def test_fix(self):
        self.assertEqual(b'person <bar@blah.nl>', fix_person_identifier(b'somebody <person <bar@blah.nl>>'))
        self.assertEqual(b'person <bar@blah.nl>', fix_person_identifier(b'person<bar@blah.nl>'))
        self.assertEqual(b'Rohan Garg <rohangarg@kubuntu.org>', fix_person_identifier(b'Rohan Garg <rohangarg@kubuntu.org'))
        self.assertRaises(ValueError, fix_person_identifier, b'person >bar@blah.nl<')