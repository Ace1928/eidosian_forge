import warnings
from datetime import datetime
from Bio import BiopythonWarning
from Bio import SeqFeature
from Bio import SeqIO
from Bio.GenBank.Scanner import _ImgtScanner
from Bio.GenBank.Scanner import EmblScanner
from Bio.GenBank.Scanner import GenBankScanner
from Bio.Seq import UndefinedSequenceError
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def _write_the_first_lines(self, record):
    """Write the ID and AC lines (PRIVATE)."""
    if '.' in record.id and record.id.rsplit('.', 1)[1].isdigit():
        version = 'SV ' + record.id.rsplit('.', 1)[1]
        accession = self._get_annotation_str(record, 'accession', record.id.rsplit('.', 1)[0], just_first=True)
    else:
        version = ''
        accession = self._get_annotation_str(record, 'accession', record.id, just_first=True)
    if ';' in accession:
        raise ValueError(f"Cannot have semi-colon in EMBL accession, '{accession}'")
    if ' ' in accession:
        raise ValueError(f"Cannot have spaces in EMBL accession, '{accession}'")
    topology = self._get_annotation_str(record, 'topology', default='')
    mol_type = record.annotations.get('molecule_type')
    if mol_type is None:
        raise ValueError('missing molecule_type in annotations')
    if mol_type not in ('DNA', 'genomic DNA', 'unassigned DNA', 'mRNA', 'RNA', 'protein'):
        warnings.warn(f'Non-standard molecule type: {mol_type}', BiopythonWarning)
    mol_type_upper = mol_type.upper()
    if 'DNA' in mol_type_upper:
        units = 'BP'
    elif 'RNA' in mol_type_upper:
        units = 'BP'
    elif 'PROTEIN' in mol_type_upper:
        mol_type = 'PROTEIN'
        units = 'AA'
    else:
        raise ValueError(f"failed to understand molecule_type '{mol_type}'")
    division = self._get_data_division(record)
    handle = self.handle
    self._write_single_line('ID', '%s; %s; %s; %s; ; %s; %i %s.' % (accession, version, topology, mol_type, division, len(record), units))
    handle.write('XX\n')
    self._write_single_line('AC', accession + ';')
    handle.write('XX\n')