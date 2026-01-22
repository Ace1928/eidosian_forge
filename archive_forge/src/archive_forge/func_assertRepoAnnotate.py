import codecs
from io import BytesIO, StringIO
from .. import annotate, tests
from .ui_testing import StringIOWithEncoding
def assertRepoAnnotate(self, expected, repo, path, revision_id):
    """Assert that the revision is properly annotated."""
    actual = list(repo.revision_tree(revision_id).annotate_iter(path))
    self.assertAnnotateEqualDiff(actual, expected)