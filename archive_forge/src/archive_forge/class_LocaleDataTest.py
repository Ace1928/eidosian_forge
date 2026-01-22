import datetime
import os
import shutil
import tempfile
import unittest
import tornado.locale
from tornado.escape import utf8, to_unicode
from tornado.util import unicode_type
class LocaleDataTest(unittest.TestCase):

    def test_non_ascii_name(self):
        name = tornado.locale.LOCALE_NAMES['es_LA']['name']
        self.assertTrue(isinstance(name, unicode_type))
        self.assertEqual(name, 'Español')
        self.assertEqual(utf8(name), b'Espa\xc3\xb1ol')