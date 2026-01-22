import os
from ....tests import TestCaseWithTransport
from ..wrapper import (quilt_applied, quilt_delete, quilt_pop_all,
from . import quilt_feature
class QuiltTests(TestCaseWithTransport):
    _test_needs_features = [quilt_feature]

    def make_empty_quilt_dir(self, path):
        source = self.make_branch_and_tree(path)
        self.build_tree([os.path.join(path, n) for n in ['patches/']])
        self.build_tree_contents([(os.path.join(path, 'patches/series'), '\n')])
        source.add(['patches', 'patches/series'])
        return source

    def test_series_all_empty(self):
        source = self.make_empty_quilt_dir('source')
        self.assertEqual([], quilt_series(source, 'patches/series'))

    def test_series_all(self):
        source = self.make_empty_quilt_dir('source')
        self.build_tree_contents([('source/patches/series', 'patch1.diff\n'), ('source/patches/patch1.diff', TRIVIAL_PATCH)])
        source.smart_add(['source'])
        self.assertEqual(['patch1.diff'], quilt_series(source, 'patches/series'))

    def test_push_all_empty(self):
        self.make_empty_quilt_dir('source')
        quilt_push_all('source', quiet=True)

    def test_pop_all_empty(self):
        self.make_empty_quilt_dir('source')
        quilt_pop_all('source', quiet=True)

    def test_applied_empty(self):
        source = self.make_empty_quilt_dir('source')
        self.build_tree_contents([('source/patches/series', 'patch1.diff\n'), ('source/patches/patch1.diff', 'foob ar')])
        self.assertEqual([], quilt_applied(source))

    def test_unapplied(self):
        self.make_empty_quilt_dir('source')
        self.build_tree_contents([('source/patches/series', 'patch1.diff\n'), ('source/patches/patch1.diff', 'foob ar')])
        self.assertEqual(['patch1.diff'], quilt_unapplied('source'))

    def test_unapplied_dir(self):
        self.make_empty_quilt_dir('source')
        self.build_tree_contents([('source/patches/series', 'debian/patch1.diff\n'), ('source/patches/debian/',), ('source/patches/debian/patch1.diff', 'foob ar')])
        self.assertEqual(['debian/patch1.diff'], quilt_unapplied('source'))

    def test_unapplied_multi(self):
        self.make_empty_quilt_dir('source')
        self.build_tree_contents([('source/patches/series', 'patch1.diff\npatch2.diff'), ('source/patches/patch1.diff', 'foob ar'), ('source/patches/patch2.diff', 'bazb ar')])
        self.assertEqual(['patch1.diff', 'patch2.diff'], quilt_unapplied('source', 'patches'))

    def test_delete(self):
        source = self.make_empty_quilt_dir('source')
        self.build_tree_contents([('source/patches/series', 'patch1.diff\npatch2.diff'), ('source/patches/patch1.diff', 'foob ar'), ('source/patches/patch2.diff', 'bazb ar')])
        quilt_delete('source', 'patch1.diff', 'patches', remove=False)
        self.assertEqual(['patch2.diff'], quilt_series(source, 'patches/series'))
        quilt_delete('source', 'patch2.diff', 'patches', remove=True)
        self.assertTrue(os.path.exists('source/patches/patch1.diff'))
        self.assertFalse(os.path.exists('source/patches/patch2.diff'))
        self.assertEqual([], quilt_series(source, 'patches/series'))