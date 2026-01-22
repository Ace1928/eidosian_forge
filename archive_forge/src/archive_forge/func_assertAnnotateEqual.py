from .. import annotate, errors, revision, tests
from ..bzr import knit
def assertAnnotateEqual(self, expected_annotation, key, exp_text=None):
    annotation, lines = self.ann.annotate(key)
    self.assertEqual(expected_annotation, annotation)
    if exp_text is None:
        record = next(self.vf.get_record_stream([key], 'unordered', True))
        exp_text = record.get_bytes_as('fulltext')
    self.assertEqualDiff(exp_text, b''.join(lines))