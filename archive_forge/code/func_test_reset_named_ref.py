from io import BytesIO
from dulwich.repo import Repo as GitRepo
from .. import tests
def test_reset_named_ref(self):
    stream = BytesIO()
    builder = tests.GitBranchBuilder(stream)
    builder.reset(b'refs/heads/branch')
    self.assertEqualDiff(b'reset refs/heads/branch\n\n', stream.getvalue())