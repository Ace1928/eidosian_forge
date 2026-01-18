from io import BytesIO
from dulwich.repo import Repo as GitRepo
from .. import tests
def test_set_symlink_newline(self):
    stream = BytesIO()
    builder = tests.GitBranchBuilder(stream)
    builder.set_symlink('foo\nbar', 'link/contents')
    self.assertEqualDiff(b'blob\nmark :1\ndata 13\nlink/contents\n', stream.getvalue())
    self.assertEqual([b'M 120000 :1 "foo\\nbar"\n'], builder.commit_info)