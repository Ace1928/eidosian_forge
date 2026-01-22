import os.path
import re
import unittest
import idna
class TestIdnaTest(unittest.TestCase):
    """Run one of the IdnaTestV2.txt test lines."""

    def __init__(self, lineno=None, fields=None):
        super().__init__()
        self.lineno = lineno
        self.fields = fields

    def id(self):
        return '{}.{}'.format(super().id(), self.lineno)

    def shortDescription(self):
        if not self.fields:
            return ''
        return 'IdnaTestV2.txt line {}: {}'.format(self.lineno, '; '.join(self.fields))

    def runTest(self):
        if not self.fields:
            return ''
        source, to_unicode, to_unicode_status, to_ascii, to_ascii_status, to_ascii_t, to_ascii_t_status = self.fields
        if source in _SKIP_TESTS:
            return
        if not to_unicode:
            to_unicode = source
        if not to_unicode_status:
            to_unicode_status = '[]'
        if not to_ascii:
            to_ascii = to_unicode
        if not to_ascii_status:
            to_ascii_status = to_unicode_status
        if not to_ascii_t:
            to_ascii_t = to_ascii
        if not to_ascii_t_status:
            to_ascii_t_status = to_ascii_status
        try:
            output = idna.decode(source, uts46=True, strict=True)
            if to_unicode_status != '[]':
                self.fail('decode() did not emit required error {} for {}'.format(to_unicode, repr(source)))
            self.assertEqual(output, to_unicode, 'unexpected decode() output')
        except (idna.IDNAError, UnicodeError, ValueError) as exc:
            if str(exc).startswith('Unknown'):
                raise unittest.SkipTest('Test requires support for a newer version of Unicode than this Python supports')
            if to_unicode_status == '[]':
                raise
        try:
            output = idna.encode(source, uts46=True, strict=True).decode('ascii')
            if to_ascii_status != '[]':
                self.fail('encode() did not emit required error {} for {}'.format(to_ascii_status, repr(source)))
            self.assertEqual(output, to_ascii, 'unexpected encode() output')
        except (idna.IDNAError, UnicodeError, ValueError) as exc:
            if str(exc).startswith('Unknown'):
                raise unittest.SkipTest('Test requires support for a newer version of Unicode than this Python supports')
            if to_ascii_status == '[]':
                raise
        try:
            output = idna.encode(source, uts46=True, strict=True, transitional=True).decode('ascii')
            if to_ascii_t_status != '[]':
                self.fail('encode(transitional=True) did not emit required error {} for {}'.format(to_ascii_t_status, repr(source)))
            self.assertEqual(output, to_ascii_t, 'unexpected encode() output')
        except (idna.IDNAError, UnicodeError, ValueError) as exc:
            if str(exc).startswith('Unknown'):
                raise unittest.SkipTest('Test requires support for a newer version of Unicode than this Python supports')
            if to_ascii_t_status == '[]':
                raise