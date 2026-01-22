from ... import errors, multiparent, tests
from .. import groupcompress, versionedfile
class Test_MPDiffGenerator(tests.TestCaseWithMemoryTransport):

    def make_vf(self):
        t = self.get_transport('')
        factory = groupcompress.make_pack_factory(True, True, 1)
        return factory(t)

    def make_three_vf(self):
        vf = self.make_vf()
        vf.add_lines((b'one',), (), [b'first\n'])
        vf.add_lines((b'two',), [(b'one',)], [b'first\n', b'second\n'])
        vf.add_lines((b'three',), [(b'one',), (b'two',)], [b'first\n', b'second\n', b'third\n'])
        return vf

    def test_finds_parents(self):
        vf = self.make_three_vf()
        gen = versionedfile._MPDiffGenerator(vf, [(b'three',)])
        needed_keys, refcount = gen._find_needed_keys()
        self.assertEqual(sorted([(b'one',), (b'two',), (b'three',)]), sorted(needed_keys))
        self.assertEqual({(b'one',): 1, (b'two',): 1}, refcount)

    def test_ignores_ghost_parents(self):
        vf = self.make_vf()
        vf.add_lines((b'two',), [(b'one',)], [b'first\n', b'second\n'])
        gen = versionedfile._MPDiffGenerator(vf, [(b'two',)])
        needed_keys, refcount = gen._find_needed_keys()
        self.assertEqual(sorted([(b'two',)]), sorted(needed_keys))
        self.assertEqual({(b'one',): 1}, refcount)
        self.assertEqual([(b'one',)], sorted(gen.ghost_parents))
        self.assertEqual([], sorted(gen.present_parents))

    def test_raises_on_ghost_keys(self):
        vf = self.make_vf()
        gen = versionedfile._MPDiffGenerator(vf, [(b'one',)])
        self.assertRaises(errors.RevisionNotPresent, gen._find_needed_keys)

    def test_refcount_multiple_children(self):
        vf = self.make_three_vf()
        gen = versionedfile._MPDiffGenerator(vf, [(b'two',), (b'three',)])
        needed_keys, refcount = gen._find_needed_keys()
        self.assertEqual(sorted([(b'one',), (b'two',), (b'three',)]), sorted(needed_keys))
        self.assertEqual({(b'one',): 2, (b'two',): 1}, refcount)
        self.assertEqual([(b'one',)], sorted(gen.present_parents))

    def test_process_contents(self):
        vf = self.make_three_vf()
        gen = versionedfile._MPDiffGenerator(vf, [(b'two',), (b'three',)])
        gen._find_needed_keys()
        self.assertEqual({(b'two',): ((b'one',),), (b'three',): ((b'one',), (b'two',))}, gen.parent_map)
        self.assertEqual({(b'one',): 2, (b'two',): 1}, gen.refcounts)
        self.assertEqual(sorted([(b'one',), (b'two',), (b'three',)]), sorted(gen.needed_keys))
        stream = vf.get_record_stream(gen.needed_keys, 'topological', True)
        record = next(stream)
        self.assertEqual((b'one',), record.key)
        gen._process_one_record(record.key, record.get_bytes_as('chunked'))
        self.assertEqual({(b'one',)}, set(gen.chunks))
        self.assertEqual({(b'one',): 2, (b'two',): 1}, gen.refcounts)
        self.assertEqual(set(), set(gen.diffs))
        record = next(stream)
        self.assertEqual((b'two',), record.key)
        gen._process_one_record(record.key, record.get_bytes_as('chunked'))
        self.assertEqual({(b'one',), (b'two',)}, set(gen.chunks))
        self.assertEqual({(b'one',): 1, (b'two',): 1}, gen.refcounts)
        self.assertEqual({(b'two',)}, set(gen.diffs))
        self.assertEqual({(b'three',): ((b'one',), (b'two',))}, gen.parent_map)
        record = next(stream)
        self.assertEqual((b'three',), record.key)
        gen._process_one_record(record.key, record.get_bytes_as('chunked'))
        self.assertEqual(set(), set(gen.chunks))
        self.assertEqual({}, gen.refcounts)
        self.assertEqual({(b'two',), (b'three',)}, set(gen.diffs))

    def test_compute_diffs(self):
        vf = self.make_three_vf()
        gen = versionedfile._MPDiffGenerator(vf, [(b'two',), (b'three',), (b'one',)])
        diffs = gen.compute_diffs()
        expected_diffs = [multiparent.MultiParent([multiparent.ParentText(0, 0, 0, 1), multiparent.NewText([b'second\n'])]), multiparent.MultiParent([multiparent.ParentText(1, 0, 0, 2), multiparent.NewText([b'third\n'])]), multiparent.MultiParent([multiparent.NewText([b'first\n'])])]
        self.assertEqual(expected_diffs, diffs)