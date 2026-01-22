from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def _parse_sequential(self, lines, seqs, names, length):
    for line in lines:
        if length == 0:
            line = line.rstrip()
            name = line[:_PHYLIP_ID_WIDTH].strip()
            seq = line[_PHYLIP_ID_WIDTH:].strip()
            names.append(name)
            seqs.append([])
        else:
            seq = line.strip()
        seq = seq.replace(' ', '')
        seqs[-1].append(seq)
        length += len(seq)
        if length == self._length_of_seqs:
            length = 0
    return length