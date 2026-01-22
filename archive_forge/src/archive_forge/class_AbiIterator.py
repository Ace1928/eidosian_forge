import datetime
import struct
import sys
from os.path import basename
from typing import List
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
class AbiIterator(SequenceIterator):
    """Parser for Abi files."""

    def __init__(self, source, trim=False):
        """Return an iterator for the Abi file format."""
        self.trim = trim
        super().__init__(source, mode='b', fmt='ABI')

    def parse(self, handle):
        """Start parsing the file, and return a SeqRecord generator."""
        marker = handle.read(4)
        if not marker:
            raise ValueError('Empty file.')
        if marker != b'ABIF':
            raise OSError(f'File should start ABIF, not {marker!r}')
        records = self.iterate(handle)
        return records

    def iterate(self, handle):
        """Parse the file and generate SeqRecord objects."""
        times = {'RUND1': '', 'RUND2': '', 'RUNT1': '', 'RUNT2': ''}
        annot = dict(zip(_EXTRACT.values(), [None] * len(_EXTRACT)))
        header = struct.unpack(_HEADFMT, handle.read(struct.calcsize(_HEADFMT)))
        sample_id = '<unknown id>'
        raw = {}
        seq = qual = None
        for tag_name, tag_number, tag_data in _abi_parse_header(header, handle):
            key = tag_name + str(tag_number)
            raw[key] = tag_data
            if key == 'PBAS2':
                seq = tag_data.decode()
            elif key == 'PCON2':
                qual = [ord(val) for val in tag_data.decode()]
            elif key == 'SMPL1':
                sample_id = _get_string_tag(tag_data)
            elif key in times:
                times[key] = tag_data
            elif key in _EXTRACT:
                annot[_EXTRACT[key]] = tag_data
        annot['run_start'] = f'{times['RUND1']} {times['RUNT1']}'
        annot['run_finish'] = f'{times['RUND2']} {times['RUNT2']}'
        annot['abif_raw'] = raw
        is_fsa_file = all((tn not in raw for tn in ('PBAS1', 'PBAS2')))
        if is_fsa_file:
            try:
                file_name = basename(handle.name).replace('.fsa', '')
            except AttributeError:
                file_name = ''
            sample_id = _get_string_tag(raw.get('LIMS1'), sample_id)
            description = _get_string_tag(raw.get('CTID1'), '<unknown description>')
            record = SeqRecord(Seq(''), id=sample_id, name=file_name, description=description, annotations=annot)
        else:
            try:
                file_name = basename(handle.name).replace('.ab1', '')
            except AttributeError:
                file_name = ''
            record = SeqRecord(Seq(seq), id=sample_id, name=file_name, description='', annotations=annot)
        if qual:
            record.letter_annotations['phred_quality'] = qual
        elif not is_fsa_file and (not qual) and self.trim:
            raise ValueError("The 'abi-trim' format can not be used for files without quality values.")
        if self.trim and (not is_fsa_file):
            record = _abi_trim(record)
        record.annotations['molecule_type'] = 'DNA'
        yield record