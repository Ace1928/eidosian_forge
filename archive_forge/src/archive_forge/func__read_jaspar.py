from Bio.Seq import Seq
import re
import math
from Bio import motifs
from Bio import Align
def _read_jaspar(handle):
    """Read motifs from a JASPAR formatted file (PRIVATE).

    Format is one or more records of the form, e.g.::

      - JASPAR 2010 matrix_only format::

                >MA0001.1 AGL3
                A  [ 0  3 79 40 66 48 65 11 65  0 ]
                C  [94 75  4  3  1  2  5  2  3  3 ]
                G  [ 1  0  3  4  1  0  5  3 28 88 ]
                T  [ 2 19 11 50 29 47 22 81  1  6 ]

      - JASPAR 2010-2014 PFMs format::

                >MA0001.1 AGL3
                0	3	79	40	66	48	65	11	65	0
                94	75	4	3	1	2	5	2	3	3
                1	0	3	4	1	0	5	3	28	88
                2	19	11	50	29	47	22	81	1	6

    """
    alphabet = 'ACGT'
    counts = {}
    record = Record()
    head_pat = re.compile('^>\\s*(\\S+)(\\s+(\\S+))?')
    row_pat_long = re.compile('\\s*([ACGT])\\s*\\[\\s*(.*)\\s*\\]')
    row_pat_short = re.compile('\\s*(.+)\\s*')
    identifier = None
    name = None
    row_count = 0
    nucleotides = ['A', 'C', 'G', 'T']
    for line in handle:
        line = line.strip()
        head_match = head_pat.match(line)
        row_match_long = row_pat_long.match(line)
        row_match_short = row_pat_short.match(line)
        if head_match:
            identifier = head_match.group(1)
            if head_match.group(3):
                name = head_match.group(3)
            else:
                name = identifier
        elif row_match_long:
            letter, counts_str = row_match_long.group(1, 2)
            words = counts_str.split()
            counts[letter] = [float(x) for x in words]
            row_count += 1
            if row_count == 4:
                record.append(Motif(identifier, name, alphabet=alphabet, counts=counts))
                identifier = None
                name = None
                counts = {}
                row_count = 0
        elif row_match_short:
            words = row_match_short.group(1).split()
            counts[nucleotides[row_count]] = [float(x) for x in words]
            row_count += 1
            if row_count == 4:
                record.append(Motif(identifier, name, alphabet=alphabet, counts=counts))
                identifier = None
                name = None
                counts = {}
                row_count = 0
    return record