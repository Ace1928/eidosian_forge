from the Biopython unit tests:
from typing import Iterator
from Bio.SeqRecord import SeqRecord
from Bio.Sequencing import Phd
from .QualityIO import _get_phred_quality
from .Interfaces import SequenceWriter
from .Interfaces import _TextIOSource
from .Interfaces import _IOSource
class PhdWriter(SequenceWriter):
    """Class to write Phd format files."""

    def __init__(self, handle: _IOSource) -> None:
        """Initialize the class."""
        super().__init__(handle)

    def write_record(self, record):
        """Write a single Phd record to the file."""
        assert record.seq, 'No sequence present in SeqRecord'
        phred_qualities = _get_phred_quality(record)
        peak_locations = record.letter_annotations.get('peak_location')
        if len(record.seq) != len(phred_qualities):
            raise ValueError('Number of phd quality scores does not match length of sequence')
        if peak_locations:
            if len(record.seq) != len(peak_locations):
                raise ValueError('Number of peak location scores does not match length of sequence')
        if None in phred_qualities:
            raise ValueError('A quality value of None was found')
        if record.description.startswith(f'{record.id} '):
            title = record.description
        else:
            title = f'{record.id} {record.description}'
        self.handle.write(f'BEGIN_SEQUENCE {self.clean(title)}\nBEGIN_COMMENT\n')
        for annot in [k.lower() for k in Phd.CKEYWORDS]:
            value = None
            if annot == 'trim':
                if record.annotations.get('trim'):
                    value = '%s %s %.4f' % record.annotations['trim']
            elif annot == 'trace_peak_area_ratio':
                if record.annotations.get('trace_peak_area_ratio'):
                    value = f'{record.annotations['trace_peak_area_ratio']:.4f}'
            else:
                value = record.annotations.get(annot)
            if value or value == 0:
                self.handle.write(f'{annot.upper()}: {value}\n')
        self.handle.write('END_COMMENT\nBEGIN_DNA\n')
        for i, site in enumerate(record.seq):
            if peak_locations:
                self.handle.write('%s %i %i\n' % (site, round(phred_qualities[i]), peak_locations[i]))
            else:
                self.handle.write('%s %i\n' % (site, round(phred_qualities[i])))
        self.handle.write('END_DNA\nEND_SEQUENCE\n')