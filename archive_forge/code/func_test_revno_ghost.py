import os
from breezy import tests
from breezy.bzr.tests.matchers import ContainsNoVfsCalls
from breezy.errors import NoSuchRevision
def test_revno_ghost(self):
    builder = self.make_branch_builder('branch')
    builder.start_series()
    revid = builder.build_snapshot([b'aghost'], [('add', ('', b'root-id', 'directory', None)), ('add', ('file', b'file-id', 'file', b'content\n'))], revision_id=b'A-id', allow_leftmost_as_ghost=True)
    builder.finish_series()
    b = builder.get_branch()

    def revision_id_to_revno(s, r):
        raise NoSuchRevision(s, r)
    self.overrideAttr(type(b), 'revision_id_to_dotted_revno', revision_id_to_revno)
    self.overrideAttr(type(b), 'revision_id_to_revno', revision_id_to_revno)
    out, err = self.run_bzr('revno branch')
    self.assertEqual('', err)
    self.assertEqual('???\n', out)