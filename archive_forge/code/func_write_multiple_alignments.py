from abc import ABC
from abc import abstractmethod
from typing import Optional
from Bio import StreamModeError
from Bio.Align import AlignmentsAbstractBaseClass
def write_multiple_alignments(self, stream, alignments):
    """Write alignments to the output file, and return the number of alignments.

        alignments - A list or iterator returning Alignment objects
        stream     - Output file stream.
        """
    count = 0
    for alignment in alignments:
        line = self.format_alignment(alignment)
        stream.write(line)
        count += 1
    return count