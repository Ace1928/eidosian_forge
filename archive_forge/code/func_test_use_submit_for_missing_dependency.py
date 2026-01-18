import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
def test_use_submit_for_missing_dependency(self):
    tree_a, tree_b, branch_c = self.make_trees()
    branch_c.pull(tree_a.branch)
    self.build_tree_contents([('tree_a/file', b'content_q\ncontent_r\n')])
    tree_a.commit('rev3a', rev_id=b'rev3a')
    md = self.from_objects(tree_a.branch.repository, b'rev3a', 500, 36, branch_c.base, base_revision_id=b'rev2a')
    revision = md.install_revisions(tree_b.branch.repository)