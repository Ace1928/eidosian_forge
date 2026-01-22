import shlex
import itertools
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq, reverse_complement
from Bio.SeqRecord import SeqRecord
def _format_score_line(self, alignment, annotations):
    try:
        score = alignment.score
    except AttributeError:
        line = 'a'
    else:
        line = f'a score={score:.6f}'
    value = annotations.get('pass')
    if value is not None:
        line += f' pass={value}'
    return line + '\n'