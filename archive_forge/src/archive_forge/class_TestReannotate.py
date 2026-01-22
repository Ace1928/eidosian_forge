import codecs
from io import BytesIO, StringIO
from .. import annotate, tests
from .ui_testing import StringIOWithEncoding
class TestReannotate(tests.TestCase):

    def annotateEqual(self, expected, parents, newlines, revision_id, blocks=None):
        annotate_list = list(annotate.reannotate(parents, newlines, revision_id, blocks))
        self.assertEqual(len(expected), len(annotate_list))
        for e, a in zip(expected, annotate_list):
            self.assertEqual(e, a)

    def test_reannotate(self):
        self.annotateEqual(parent_1, [parent_1], new_1, b'blahblah')
        self.annotateEqual(expected_2_1, [parent_2], new_1, b'blahblah')
        self.annotateEqual(expected_1_2_2, [parent_1, parent_2], new_2, b'blahblah')

    def test_reannotate_no_parents(self):
        self.annotateEqual(expected_1, [], new_1, b'blahblah')

    def test_reannotate_left_matching_blocks(self):
        """Ensure that left_matching_blocks has an impact.

        In this case, the annotation is ambiguous, so the hint isn't actually
        lying.
        """
        parent = [(b'rev1', b'a\n')]
        new_text = [b'a\n', b'a\n']
        blocks = [(0, 0, 1), (1, 2, 0)]
        self.annotateEqual([(b'rev1', b'a\n'), (b'rev2', b'a\n')], [parent], new_text, b'rev2', blocks)
        blocks = [(0, 1, 1), (1, 2, 0)]
        self.annotateEqual([(b'rev2', b'a\n'), (b'rev1', b'a\n')], [parent], new_text, b'rev2', blocks)