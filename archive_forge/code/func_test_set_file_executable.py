from io import BytesIO
from dulwich.repo import Repo as GitRepo
from .. import tests
def test_set_file_executable(self):
    stream = BytesIO()
    builder = tests.GitBranchBuilder(stream)
    builder.set_file('fµ/bar', b'contents\nbar\n', True)
    self.assertEqualDiff(b'blob\nmark :1\ndata 13\ncontents\nbar\n\n', stream.getvalue())
    self.assertEqual([b'M 100755 :1 f\xc2\xb5/bar\n'], builder.commit_info)