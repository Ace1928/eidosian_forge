from ... import tests
from .. import generate_ids
class TestFileIds(tests.TestCase):
    """Test functions which generate file ids"""

    def assertGenFileId(self, regex, filename):
        """gen_file_id should create a file id matching the regex.

        The file id should be ascii, and should be an 8-bit string
        """
        file_id = generate_ids.gen_file_id(filename)
        self.assertContainsRe(file_id, b'^' + regex + b'$')
        self.assertIsInstance(file_id, bytes)
        file_id.decode('ascii')

    def test_gen_file_id(self):
        gen_file_id = generate_ids.gen_file_id
        self.assertStartsWith(gen_file_id('bar'), b'bar-')
        self.assertStartsWith(gen_file_id('Mwoo oof\t m'), b'mwoooofm-')
        self.assertStartsWith(gen_file_id('..gam.py'), b'gam.py-')
        self.assertStartsWith(gen_file_id('..Mwoo oof\t m'), b'mwoooofm-')
        self.assertStartsWith(gen_file_id('åµ.txt'), b'txt-')
        fid = gen_file_id('A' * 50 + '.txt')
        self.assertStartsWith(fid, b'a' * 20 + b'-')
        self.assertTrue(len(fid) < 60)
        fid = gen_file_id('åµ..aBcd\tefGhijKLMnop\tqrstuvwxyz')
        self.assertStartsWith(fid, b'abcdefghijklmnopqrst-')
        self.assertTrue(len(fid) < 60)

    def test_file_ids_are_ascii(self):
        tail = b'-\\d{14}-[a-z0-9]{16}-\\d+'
        self.assertGenFileId(b'foo' + tail, 'foo')
        self.assertGenFileId(b'foo' + tail, 'foo')
        self.assertGenFileId(b'bar' + tail, 'bar')
        self.assertGenFileId(b'br' + tail, 'bår')

    def test__next_id_suffix_sets_suffix(self):
        generate_ids._gen_file_id_suffix = None
        generate_ids._next_id_suffix()
        self.assertNotEqual(None, generate_ids._gen_file_id_suffix)

    def test__next_id_suffix_increments(self):
        generate_ids._gen_file_id_suffix = b'foo-'
        generate_ids._gen_file_id_serial = 1
        try:
            self.assertEqual(b'foo-2', generate_ids._next_id_suffix())
            self.assertEqual(b'foo-3', generate_ids._next_id_suffix())
            self.assertEqual(b'foo-4', generate_ids._next_id_suffix())
            self.assertEqual(b'foo-5', generate_ids._next_id_suffix())
            self.assertEqual(b'foo-6', generate_ids._next_id_suffix())
            self.assertEqual(b'foo-7', generate_ids._next_id_suffix())
            self.assertEqual(b'foo-8', generate_ids._next_id_suffix())
            self.assertEqual(b'foo-9', generate_ids._next_id_suffix())
            self.assertEqual(b'foo-10', generate_ids._next_id_suffix())
        finally:
            generate_ids._gen_file_id_suffix = None
            generate_ids._gen_file_id_serial = 0

    def test_gen_root_id(self):
        root_id = generate_ids.gen_root_id()
        self.assertStartsWith(root_id, b'tree_root-')