import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
class TestMergeDirective1(tests.TestCase, TestMergeDirective):
    """Test merge directive format 1"""
    INPUT1 = INPUT1
    OUTPUT1 = OUTPUT1
    OUTPUT2 = OUTPUT2

    def make_merge_directive(self, revision_id, testament_sha1, time, timezone, target_branch, patch=None, patch_type=None, source_branch=None, message=None):
        return merge_directive.MergeDirective(revision_id, testament_sha1, time, timezone, target_branch, patch, patch_type, source_branch, message)

    @staticmethod
    def set_bundle(md, value):
        md.patch = value

    def test_require_patch(self):
        time = 500.0
        timezone = 120
        self.assertRaises(errors.PatchMissing, merge_directive.MergeDirective, b'example:', b'sha', time, timezone, 'http://example.com', patch_type='bundle')
        md = merge_directive.MergeDirective(b'example:', b'sha1', time, timezone, 'http://example.com', source_branch='http://example.org', patch=b'', patch_type='diff')
        self.assertEqual(md.patch, b'')