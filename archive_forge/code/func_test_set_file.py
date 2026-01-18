from io import BytesIO
from dulwich.repo import Repo as GitRepo
from .. import tests
def test_set_file(self):
    stream = BytesIO()
    builder = tests.GitBranchBuilder(stream)
    builder.set_file('foobar', b'foo\nbar\n', False)
    self.assertEqualDiff(b'blob\nmark :1\ndata 8\nfoo\nbar\n\n', stream.getvalue())
    self.assertEqual([b'M 100644 :1 foobar\n'], builder.commit_info)