import numpy as np
from Bio.Align import Alignment, Alignments
from Bio.Align import bigbed, psl
from Bio.Align.bigbed import AutoSQLTable, Field
from Bio.Seq import Seq, reverse_complement, UndefinedSequenceError
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, Location
from Bio.SeqIO.InsdcIO import _insdc_location_string
def _analyze_fields(self, fields, fieldCount, definedFieldCount):
    names = ('chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'thickStart', 'thickEnd', 'reserved', 'blockCount', 'blockSizes', 'chromStarts', 'oChromStart', 'oChromEnd', 'oStrand', 'oChromSize', 'oChromStarts', 'oSequence', 'oCDS', 'chromSize', 'match', 'misMatch', 'repMatch', 'nCount', 'seqType')
    for i, name in enumerate(names):
        if name != fields[i].name:
            raise ValueError("Expected field name '%s'; found '%s'" % (name, fields[i].name))