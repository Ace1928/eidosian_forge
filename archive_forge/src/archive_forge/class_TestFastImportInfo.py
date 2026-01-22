from io import BytesIO
from unittest import TestCase
from fastimport import (
from fastimport.processors import (
class TestFastImportInfo(TestCase):

    def test_simple(self):
        stream = BytesIO(simple_fast_import_stream)
        outf = StringIO()
        proc = info_processor.InfoProcessor(outf=outf)
        p = parser.ImportParser(stream)
        proc.process(p.iter_commands)
        self.maxDiff = None
        self.assertEqual(outf.getvalue(), 'Command counts:\n\t0\tblob\n\t0\tcheckpoint\n\t1\tcommit\n\t0\tfeature\n\t0\tprogress\n\t0\treset\n\t0\ttag\nFile command counts:\n\t0\tfilemodify\n\t0\tfiledelete\n\t0\tfilecopy\n\t0\tfilerename\n\t0\tfiledeleteall\nParent counts:\n\t1\tparents-0\n\t0\ttotal revisions merged\nCommit analysis:\n\tno\tblobs referenced by SHA\n\tno\texecutables\n\tno\tseparate authors found\n\tno\tsymlinks\nHead analysis:\n\t:1\trefs/heads/master\nMerges:\n')