from io import BytesIO
from dulwich.repo import Repo as GitRepo
from .. import tests
def test_reset_revision(self):
    stream = BytesIO()
    builder = tests.GitBranchBuilder(stream)
    builder.reset(mark=b'123')
    self.assertEqualDiff(b'reset refs/heads/master\nfrom :123\n\n', stream.getvalue())