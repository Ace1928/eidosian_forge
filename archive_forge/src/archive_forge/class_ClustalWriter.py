from Bio.Align import MultipleSeqAlignment
from Bio.AlignIO.Interfaces import AlignmentIterator
from Bio.AlignIO.Interfaces import SequentialAlignmentWriter
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
class ClustalWriter(SequentialAlignmentWriter):
    """Clustalw alignment writer."""

    def write_alignment(self, alignment):
        """Use this to write (another) single alignment to an open file."""
        if len(alignment) == 0:
            raise ValueError('Must have at least one sequence')
        if alignment.get_alignment_length() == 0:
            raise ValueError('Non-empty sequences are required')
        try:
            version = str(alignment._version)
        except AttributeError:
            version = ''
        if not version:
            version = '1.81'
        if version.startswith('2.'):
            output = f'CLUSTAL {version} multiple sequence alignment\n\n\n'
        else:
            output = f'CLUSTAL X ({version}) multiple sequence alignment\n\n\n'
        cur_char = 0
        max_length = len(alignment[0])
        if max_length <= 0:
            raise ValueError('Non-empty sequences are required')
        if 'clustal_consensus' in alignment.column_annotations:
            star_info = alignment.column_annotations['clustal_consensus']
        else:
            try:
                star_info = alignment._star_info
            except AttributeError:
                star_info = None
        while cur_char != max_length:
            if cur_char + 50 > max_length:
                show_num = max_length - cur_char
            else:
                show_num = 50
            for record in alignment:
                line = record.id[0:30].replace(' ', '_').ljust(36)
                line += str(record.seq[cur_char:cur_char + show_num])
                output += line + '\n'
            if star_info:
                output += ' ' * 36 + star_info[cur_char:cur_char + show_num] + '\n'
            output += '\n'
            cur_char += show_num
        self.handle.write(output + '\n')