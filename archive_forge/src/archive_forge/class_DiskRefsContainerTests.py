import os
import sys
import tempfile
from io import BytesIO
from typing import ClassVar, Dict
from dulwich import errors
from dulwich.tests import SkipTest, TestCase
from ..file import GitFile
from ..objects import ZERO_SHA
from ..refs import (
from ..repo import Repo
from .utils import open_repo, tear_down_repo
class DiskRefsContainerTests(RefsContainerTests, TestCase):

    def setUp(self):
        TestCase.setUp(self)
        self._repo = open_repo('refs.git')
        self.addCleanup(tear_down_repo, self._repo)
        self._refs = self._repo.refs

    def test_get_packed_refs(self):
        self.assertEqual({b'refs/heads/packed': b'42d06bd4b77fed026b154d16493e5deab78f02ec', b'refs/tags/refs-0.1': b'df6800012397fb85c56e7418dd4eb9405dee075c'}, self._refs.get_packed_refs())

    def test_get_peeled_not_packed(self):
        self.assertEqual(None, self._refs.get_peeled(b'refs/tags/refs-0.2'))
        self.assertEqual(b'3ec9c43c84ff242e3ef4a9fc5bc111fd780a76a8', self._refs[b'refs/tags/refs-0.2'])
        self.assertEqual(self._refs[b'refs/heads/packed'], self._refs.get_peeled(b'refs/heads/packed'))
        self.assertEqual(b'42d06bd4b77fed026b154d16493e5deab78f02ec', self._refs.get_peeled(b'refs/tags/refs-0.1'))

    def test_setitem(self):
        RefsContainerTests.test_setitem(self)
        path = os.path.join(self._refs.path, b'refs', b'some', b'ref')
        with open(path, 'rb') as f:
            self.assertEqual(b'42d06bd4b77fed026b154d16493e5deab78f02ec', f.read()[:40])
        self.assertRaises(OSError, self._refs.__setitem__, b'refs/some/ref/sub', b'42d06bd4b77fed026b154d16493e5deab78f02ec')

    def test_delete_refs_container(self):
        self._refs[b'refs/heads/blah'] = b'42d06bd4b77fed026b154d16493e5deab78f02ec'
        for ref in self._refs.allkeys():
            del self._refs[ref]
        self.assertTrue(os.path.exists(os.path.join(self._refs.path, b'refs')))

    def test_setitem_packed(self):
        with open(os.path.join(self._refs.path, b'packed-refs'), 'w') as f:
            f.write('# pack-refs with: peeled fully-peeled sorted \n')
            f.write('42d06bd4b77fed026b154d16493e5deab78f02ec refs/heads/packed\n')
        self._refs[b'refs/heads/packed'] = b'3ec9c43c84ff242e3ef4a9fc5bc111fd780a76a8'
        packed_ref_path = os.path.join(self._refs.path, b'refs', b'heads', b'packed')
        with open(packed_ref_path, 'rb') as f:
            self.assertEqual(b'3ec9c43c84ff242e3ef4a9fc5bc111fd780a76a8', f.read()[:40])
        self.assertRaises(OSError, self._refs.__setitem__, b'refs/heads/packed/sub', b'42d06bd4b77fed026b154d16493e5deab78f02ec')
        self.assertEqual({b'refs/heads/packed': b'42d06bd4b77fed026b154d16493e5deab78f02ec'}, self._refs.get_packed_refs())

    def test_add_packed_refs(self):
        self._refs[b'refs/heads/packed'] = b'3ec9c43c84ff242e3ef4a9fc5bc111fd780a76a8'
        packed_ref_path = os.path.join(self._refs.path, b'refs', b'heads', b'packed')
        self.assertTrue(os.path.exists(packed_ref_path))
        packed_refs_file_path = os.path.join(self._refs.path, b'packed-refs')
        self._refs.add_packed_refs({b'refs/heads/packed': b'42d06bd4b77fed026b154d16493e5deab78f02ec'})
        self.assertFalse(os.path.exists(packed_ref_path))
        self._refs.add_packed_refs({b'refs/heads/packed': None})
        self.assertFalse(os.path.exists(packed_ref_path))
        self.assertRaises(KeyError, self._refs.__getitem__, b'refs/heads/packed')
        self.assertRaises(ValueError, self._refs.add_packed_refs, {b'HEAD': '02ac81614bcdbd585a37b4b0edf8cb8a'})
        self._refs.add_packed_refs({ref: None for ref in self._refs.get_packed_refs()})
        self.assertEqual({}, self._refs.get_packed_refs())
        os.remove(packed_refs_file_path)
        self._refs.add_packed_refs({})
        self.assertFalse(os.path.exists(packed_refs_file_path))

    def test_setitem_symbolic(self):
        ones = b'1' * 40
        self._refs[b'HEAD'] = ones
        self.assertEqual(ones, self._refs[b'HEAD'])
        f = open(os.path.join(self._refs.path, b'HEAD'), 'rb')
        v = next(iter(f)).rstrip(b'\n\r')
        f.close()
        self.assertEqual(b'ref: refs/heads/master', v)
        f = open(os.path.join(self._refs.path, b'refs', b'heads', b'master'), 'rb')
        self.assertEqual(ones, f.read()[:40])
        f.close()

    def test_set_if_equals(self):
        RefsContainerTests.test_set_if_equals(self)
        self.assertEqual(b'9' * 40, self._refs[b'refs/heads/master'])
        self.assertFalse(os.path.exists(os.path.join(self._refs.path, b'refs', b'heads', b'master.lock')))
        self.assertFalse(os.path.exists(os.path.join(self._refs.path, b'HEAD.lock')))

    def test_add_if_new_packed(self):
        self.assertFalse(self._refs.add_if_new(b'refs/tags/refs-0.1', b'9' * 40))
        self.assertEqual(b'df6800012397fb85c56e7418dd4eb9405dee075c', self._refs[b'refs/tags/refs-0.1'])

    def test_add_if_new_symbolic(self):
        repo_dir = os.path.join(tempfile.mkdtemp(), 'test')
        os.makedirs(repo_dir)
        repo = Repo.init(repo_dir)
        self.addCleanup(tear_down_repo, repo)
        refs = repo.refs
        nines = b'9' * 40
        self.assertEqual(b'ref: refs/heads/master', refs.read_ref(b'HEAD'))
        self.assertNotIn(b'refs/heads/master', refs)
        self.assertTrue(refs.add_if_new(b'HEAD', nines))
        self.assertEqual(b'ref: refs/heads/master', refs.read_ref(b'HEAD'))
        self.assertEqual(nines, refs[b'HEAD'])
        self.assertEqual(nines, refs[b'refs/heads/master'])
        self.assertFalse(refs.add_if_new(b'HEAD', b'1' * 40))
        self.assertEqual(nines, refs[b'HEAD'])
        self.assertEqual(nines, refs[b'refs/heads/master'])

    def test_follow(self):
        self.assertEqual(([b'HEAD', b'refs/heads/master'], b'42d06bd4b77fed026b154d16493e5deab78f02ec'), self._refs.follow(b'HEAD'))
        self.assertEqual(([b'refs/heads/master'], b'42d06bd4b77fed026b154d16493e5deab78f02ec'), self._refs.follow(b'refs/heads/master'))
        self.assertRaises(SymrefLoop, self._refs.follow, b'refs/heads/loop')

    def test_set_overwrite_loop(self):
        self.assertRaises(SymrefLoop, self._refs.follow, b'refs/heads/loop')
        self._refs[b'refs/heads/loop'] = b'42d06bd4b77fed026b154d16493e5deab78f02ec'
        self.assertEqual(([b'refs/heads/loop'], b'42d06bd4b77fed026b154d16493e5deab78f02ec'), self._refs.follow(b'refs/heads/loop'))

    def test_delitem(self):
        RefsContainerTests.test_delitem(self)
        ref_file = os.path.join(self._refs.path, b'refs', b'heads', b'master')
        self.assertFalse(os.path.exists(ref_file))
        self.assertNotIn(b'refs/heads/master', self._refs.get_packed_refs())

    def test_delitem_symbolic(self):
        self.assertEqual(b'ref: refs/heads/master', self._refs.read_loose_ref(b'HEAD'))
        del self._refs[b'HEAD']
        self.assertRaises(KeyError, lambda: self._refs[b'HEAD'])
        self.assertEqual(b'42d06bd4b77fed026b154d16493e5deab78f02ec', self._refs[b'refs/heads/master'])
        self.assertFalse(os.path.exists(os.path.join(self._refs.path, b'HEAD')))

    def test_remove_if_equals_symref(self):
        self.assertFalse(self._refs.remove_if_equals(b'HEAD', b'42d06bd4b77fed026b154d16493e5deab78f02ec'))
        self.assertTrue(self._refs.remove_if_equals(b'refs/heads/master', b'42d06bd4b77fed026b154d16493e5deab78f02ec'))
        self.assertRaises(KeyError, lambda: self._refs[b'refs/heads/master'])
        self.assertRaises(KeyError, lambda: self._refs[b'HEAD'])
        self.assertEqual(b'ref: refs/heads/master', self._refs.read_loose_ref(b'HEAD'))
        self.assertFalse(os.path.exists(os.path.join(self._refs.path, b'refs', b'heads', b'master.lock')))
        self.assertFalse(os.path.exists(os.path.join(self._refs.path, b'HEAD.lock')))

    def test_remove_packed_without_peeled(self):
        refs_file = os.path.join(self._repo.path, 'packed-refs')
        f = GitFile(refs_file)
        refs_data = f.read()
        f.close()
        f = GitFile(refs_file, 'wb')
        f.write(b'\n'.join((line for line in refs_data.split(b'\n') if not line or line[0] not in b'#^')))
        f.close()
        self._repo = Repo(self._repo.path)
        refs = self._repo.refs
        self.assertTrue(refs.remove_if_equals(b'refs/heads/packed', b'42d06bd4b77fed026b154d16493e5deab78f02ec'))

    def test_remove_if_equals_packed(self):
        self.assertEqual(b'df6800012397fb85c56e7418dd4eb9405dee075c', self._refs[b'refs/tags/refs-0.1'])
        self.assertTrue(self._refs.remove_if_equals(b'refs/tags/refs-0.1', b'df6800012397fb85c56e7418dd4eb9405dee075c'))
        self.assertRaises(KeyError, lambda: self._refs[b'refs/tags/refs-0.1'])

    def test_remove_parent(self):
        self._refs[b'refs/heads/foo/bar'] = b'df6800012397fb85c56e7418dd4eb9405dee075c'
        del self._refs[b'refs/heads/foo/bar']
        ref_file = os.path.join(self._refs.path, b'refs', b'heads', b'foo', b'bar')
        self.assertFalse(os.path.exists(ref_file))
        ref_file = os.path.join(self._refs.path, b'refs', b'heads', b'foo')
        self.assertFalse(os.path.exists(ref_file))
        ref_file = os.path.join(self._refs.path, b'refs', b'heads')
        self.assertTrue(os.path.exists(ref_file))
        self._refs[b'refs/heads/foo'] = b'df6800012397fb85c56e7418dd4eb9405dee075c'

    def test_read_ref(self):
        self.assertEqual(b'ref: refs/heads/master', self._refs.read_ref(b'HEAD'))
        self.assertEqual(b'42d06bd4b77fed026b154d16493e5deab78f02ec', self._refs.read_ref(b'refs/heads/packed'))
        self.assertEqual(None, self._refs.read_ref(b'nonexistent'))

    def test_read_loose_ref(self):
        self._refs[b'refs/heads/foo'] = b'df6800012397fb85c56e7418dd4eb9405dee075c'
        self.assertEqual(None, self._refs.read_ref(b'refs/heads/foo/bar'))

    def test_non_ascii(self):
        try:
            encoded_ref = os.fsencode('refs/tags/schön')
        except UnicodeEncodeError as exc:
            raise SkipTest("filesystem encoding doesn't support special character") from exc
        p = os.path.join(os.fsencode(self._repo.path), encoded_ref)
        with open(p, 'w') as f:
            f.write('00' * 20)
        expected_refs = dict(_TEST_REFS)
        expected_refs[encoded_ref] = b'00' * 20
        del expected_refs[b'refs/heads/loop']
        self.assertEqual(expected_refs, self._repo.get_refs())

    def test_cyrillic(self):
        if sys.platform in ('darwin', 'win32'):
            raise SkipTest("filesystem encoding doesn't support arbitrary bytes")
        name = b'\xcd\xee\xe2\xe0\xff\xe2\xe5\xf2\xea\xe01'
        encoded_ref = b'refs/heads/' + name
        with open(os.path.join(os.fsencode(self._repo.path), encoded_ref), 'w') as f:
            f.write('00' * 20)
        expected_refs = set(_TEST_REFS.keys())
        expected_refs.add(encoded_ref)
        self.assertEqual(expected_refs, set(self._repo.refs.allkeys()))
        self.assertEqual({r[len(b'refs/'):] for r in expected_refs if r.startswith(b'refs/')}, set(self._repo.refs.subkeys(b'refs/')))
        expected_refs.remove(b'refs/heads/loop')
        expected_refs.add(b'HEAD')
        self.assertEqual(expected_refs, set(self._repo.get_refs().keys()))