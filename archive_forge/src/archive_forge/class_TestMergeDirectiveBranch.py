import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
class TestMergeDirectiveBranch:

    def make_trees(self):
        tree_a = self.make_branch_and_tree('tree_a')
        tree_a.branch.get_config_stack().set('email', 'J. Random Hacker <jrandom@example.com>')
        self.build_tree_contents([('tree_a/file', b'content_a\ncontent_b\n'), ('tree_a/file_2', b'content_x\rcontent_y\r')])
        tree_a.add(['file', 'file_2'])
        tree_a.commit('message', rev_id=b'rev1')
        tree_b = tree_a.controldir.sprout('tree_b').open_workingtree()
        branch_c = tree_a.controldir.sprout('branch_c').open_branch()
        tree_b.commit('message', rev_id=b'rev2b')
        self.build_tree_contents([('tree_a/file', b'content_a\ncontent_c \n'), ('tree_a/file_2', b'content_x\rcontent_z\r')])
        tree_a.commit('Commit of rev2a', rev_id=b'rev2a')
        return (tree_a, tree_b, branch_c)

    def test_empty_target(self):
        tree_a, tree_b, branch_c = self.make_trees()
        tree_d = self.make_branch_and_tree('tree_d')
        md2 = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 120, tree_d.branch.base, patch_type='diff', public_branch=tree_a.branch.base)

    def test_disk_name(self):
        tree_a, tree_b, branch_c = self.make_trees()
        tree_a.branch.nick = 'fancy <name>'
        md = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 120, tree_b.branch.base)
        self.assertEqual('fancy-name-2', md.get_disk_name(tree_a.branch))

    def test_disk_name_old_revno(self):
        tree_a, tree_b, branch_c = self.make_trees()
        tree_a.branch.nick = 'fancy-name'
        md = self.from_objects(tree_a.branch.repository, b'rev1', 500, 120, tree_b.branch.base)
        self.assertEqual('fancy-name-1', md.get_disk_name(tree_a.branch))

    def test_generate_patch(self):
        tree_a, tree_b, branch_c = self.make_trees()
        md2 = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 120, tree_b.branch.base, patch_type='diff', public_branch=tree_a.branch.base)
        self.assertNotContainsRe(md2.patch, b'Bazaar revision bundle')
        self.assertContainsRe(md2.patch, b'\\+content_c')
        self.assertNotContainsRe(md2.patch, b'\\+\\+\\+ b/')
        self.assertContainsRe(md2.patch, b'\\+\\+\\+ file')

    def test_public_branch(self):
        tree_a, tree_b, branch_c = self.make_trees()
        self.assertRaises(errors.PublicBranchOutOfDate, self.from_objects, tree_a.branch.repository, b'rev2a', 500, 144, tree_b.branch.base, public_branch=branch_c.base, patch_type='diff')
        self.assertRaises(errors.PublicBranchOutOfDate, self.from_objects, tree_a.branch.repository, b'rev2a', 500, 144, tree_b.branch.base, public_branch=branch_c.base, patch_type=None)
        md1 = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 144, tree_b.branch.base, public_branch=branch_c.base)
        self.assertEqual(md1.source_branch, branch_c.base)
        branch_c.pull(tree_a.branch)
        md3 = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 144, tree_b.branch.base, patch_type=None, public_branch=branch_c.base)

    def test_use_public_submit_branch(self):
        tree_a, tree_b, branch_c = self.make_trees()
        branch_c.pull(tree_a.branch)
        md = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 144, tree_b.branch.base, patch_type=None, public_branch=branch_c.base)
        self.assertEqual(md.target_branch, tree_b.branch.base)
        tree_b.branch.set_public_branch('http://example.com')
        md2 = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 144, tree_b.branch.base, patch_type=None, public_branch=branch_c.base)
        self.assertEqual(md2.target_branch, 'http://example.com')

    def test_message(self):
        tree_a, tree_b, branch_c = self.make_trees()
        md3 = self.from_objects(tree_a.branch.repository, b'rev1', 500, 120, tree_b.branch.base, patch_type=None, public_branch=branch_c.base, message='Merge message')
        md3.to_lines()
        self.assertIs(None, md3.patch)
        self.assertEqual('Merge message', md3.message)

    def test_generate_bundle(self):
        tree_a, tree_b, branch_c = self.make_trees()
        md1 = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 120, tree_b.branch.base, public_branch=branch_c.base)
        self.assertContainsRe(md1.get_raw_bundle(), b'Bazaar revision bundle')
        self.assertContainsRe(md1.patch, b'\\+content_c')
        self.assertNotContainsRe(md1.patch, b'\\+content_a')
        self.assertContainsRe(md1.patch, b'\\+content_c')
        self.assertNotContainsRe(md1.patch, b'\\+content_a')

    def test_broken_bundle(self):
        tree_a, tree_b, branch_c = self.make_trees()
        md1 = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 120, tree_b.branch.base, public_branch=branch_c.base)
        lines = md1.to_lines()
        lines = [l.replace(b'\n', b'\r\n') for l in lines]
        md2 = merge_directive.MergeDirective.from_lines(lines)
        self.assertEqual(b'rev2a', md2.revision_id)

    def test_signing(self):
        time = 453
        timezone = 7200

        class FakeBranch:

            def get_config_stack(self):
                return self
        md = self.make_merge_directive(b'example:', b'sha', time, timezone, 'http://example.com', source_branch='http://example.org', patch=b'booga', patch_type='diff')
        old_strategy = gpg.GPGStrategy
        gpg.GPGStrategy = gpg.LoopbackGPGStrategy
        try:
            signed = md.to_signed(FakeBranch())
        finally:
            gpg.GPGStrategy = old_strategy
        self.assertContainsRe(signed, b'^-----BEGIN PSEUDO-SIGNED CONTENT')
        self.assertContainsRe(signed, b'example.org')
        self.assertContainsRe(signed, b'booga')

    def test_email(self):
        tree_a, tree_b, branch_c = self.make_trees()
        md = self.from_objects(tree_a.branch.repository, b'rev2a', 476, 60, tree_b.branch.base, patch_type=None, public_branch=tree_a.branch.base)
        message = md.to_email('pqm@example.com', tree_a.branch)
        self.assertContainsRe(message.as_string(), self.EMAIL1)
        md.message = 'Commit of rev2a with special message'
        message = md.to_email('pqm@example.com', tree_a.branch)
        self.assertContainsRe(message.as_string(), self.EMAIL2)

    def test_install_revisions_branch(self):
        tree_a, tree_b, branch_c = self.make_trees()
        md = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 36, tree_b.branch.base, patch_type=None, public_branch=tree_a.branch.base)
        self.assertFalse(tree_b.branch.repository.has_revision(b'rev2a'))
        revision = md.install_revisions(tree_b.branch.repository)
        self.assertEqual(b'rev2a', revision)
        self.assertTrue(tree_b.branch.repository.has_revision(b'rev2a'))

    def test_get_merge_request(self):
        tree_a, tree_b, branch_c = self.make_trees()
        md = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 36, tree_b.branch.base, patch_type='bundle', public_branch=tree_a.branch.base)
        self.assertFalse(tree_b.branch.repository.has_revision(b'rev2a'))
        md.install_revisions(tree_b.branch.repository)
        base, revision, verified = md.get_merge_request(tree_b.branch.repository)
        if isinstance(md, merge_directive.MergeDirective):
            self.assertIs(None, base)
            self.assertEqual('inapplicable', verified)
        else:
            self.assertEqual(b'rev1', base)
            self.assertEqual('verified', verified)
        self.assertEqual(b'rev2a', revision)
        self.assertTrue(tree_b.branch.repository.has_revision(b'rev2a'))
        md = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 36, tree_b.branch.base, patch_type=None, public_branch=tree_a.branch.base)
        base, revision, verified = md.get_merge_request(tree_b.branch.repository)
        if isinstance(md, merge_directive.MergeDirective):
            self.assertIs(None, base)
            self.assertEqual('inapplicable', verified)
        else:
            self.assertEqual(b'rev1', base)
            self.assertEqual('inapplicable', verified)
        md = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 36, tree_b.branch.base, patch_type='diff', public_branch=tree_a.branch.base)
        base, revision, verified = md.get_merge_request(tree_b.branch.repository)
        if isinstance(md, merge_directive.MergeDirective):
            self.assertIs(None, base)
            self.assertEqual('inapplicable', verified)
        else:
            self.assertEqual(b'rev1', base)
            self.assertEqual('verified', verified)
        md.patch = b'asdf'
        base, revision, verified = md.get_merge_request(tree_b.branch.repository)
        if isinstance(md, merge_directive.MergeDirective):
            self.assertIs(None, base)
            self.assertEqual('inapplicable', verified)
        else:
            self.assertEqual(b'rev1', base)
            self.assertEqual('failed', verified)

    def test_install_revisions_bundle(self):
        tree_a, tree_b, branch_c = self.make_trees()
        md = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 36, tree_b.branch.base, patch_type='bundle', public_branch=tree_a.branch.base)
        self.assertFalse(tree_b.branch.repository.has_revision(b'rev2a'))
        revision = md.install_revisions(tree_b.branch.repository)
        self.assertEqual(b'rev2a', revision)
        self.assertTrue(tree_b.branch.repository.has_revision(b'rev2a'))

    def test_get_target_revision_nofetch(self):
        tree_a, tree_b, branch_c = self.make_trees()
        tree_b.branch.fetch(tree_a.branch)
        md = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 36, tree_b.branch.base, patch_type=None, public_branch=tree_a.branch.base)
        md.source_branch = '/dev/null'
        revision = md.install_revisions(tree_b.branch.repository)
        self.assertEqual(b'rev2a', revision)

    def test_use_submit_for_missing_dependency(self):
        tree_a, tree_b, branch_c = self.make_trees()
        branch_c.pull(tree_a.branch)
        self.build_tree_contents([('tree_a/file', b'content_q\ncontent_r\n')])
        tree_a.commit('rev3a', rev_id=b'rev3a')
        md = self.from_objects(tree_a.branch.repository, b'rev3a', 500, 36, branch_c.base, base_revision_id=b'rev2a')
        revision = md.install_revisions(tree_b.branch.repository)

    def test_handle_target_not_a_branch(self):
        tree_a, tree_b, branch_c = self.make_trees()
        branch_c.pull(tree_a.branch)
        self.build_tree_contents([('tree_a/file', b'content_q\ncontent_r\n')])
        tree_a.commit('rev3a', rev_id=b'rev3a')
        md = self.from_objects(tree_a.branch.repository, b'rev3a', 500, 36, branch_c.base, base_revision_id=b'rev2a')
        md.target_branch = self.get_url('not-a-branch')
        self.assertRaises(errors.TargetNotBranch, md.install_revisions, tree_b.branch.repository)