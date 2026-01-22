import warnings
import re
import sys
from collections import defaultdict
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonParserWarning
from typing import List
def _feed_first_line(self, consumer, line):
    """Scan over and parse GenBank LOCUS line (PRIVATE).

        This must cope with several variants, primarily the old and new column
        based standards from GenBank. Additionally EnsEMBL produces GenBank
        files where the LOCUS line is space separated rather that following
        the column based layout.

        We also try to cope with GenBank like files with partial LOCUS lines.

        As of release 229.0, the columns are no longer strictly in a given
        position. See GenBank format release notes:

            "Historically, the LOCUS line has had a fixed length and its
            elements have been presented at specific column positions...
            But with the anticipated increases in the lengths of accession
            numbers, and the advent of sequences that are gigabases long,
            maintaining the column positions will not always be possible and
            the overall length of the LOCUS line could exceed 79 characters."

        """
    if line[0:self.GENBANK_INDENT] != 'LOCUS       ':
        raise ValueError('LOCUS line does not start correctly:\n' + line)
    if line[29:33] in [' bp ', ' aa ', ' rc '] and line[55:62] == '       ':
        if line[41:42] != ' ':
            raise ValueError('LOCUS line does not contain space at position 42:\n' + line)
        if line[42:51].strip() not in ['', 'linear', 'circular']:
            raise ValueError('LOCUS line does not contain valid entry (linear, circular, ...):\n' + line)
        if line[51:52] != ' ':
            raise ValueError('LOCUS line does not contain space at position 52:\n' + line)
        if line[62:73].strip():
            if line[64:65] != '-':
                raise ValueError('LOCUS line does not contain - at position 65 in date:\n' + line)
            if line[68:69] != '-':
                raise ValueError('LOCUS line does not contain - at position 69 in date:\n' + line)
        name_and_length_str = line[self.GENBANK_INDENT:29]
        while '  ' in name_and_length_str:
            name_and_length_str = name_and_length_str.replace('  ', ' ')
        name_and_length = name_and_length_str.split(' ')
        if len(name_and_length) > 2:
            raise ValueError('Cannot parse the name and length in the LOCUS line:\n' + line)
        if len(name_and_length) == 1:
            raise ValueError('Name and length collide in the LOCUS line:\n' + line)
        name, length = name_and_length
        if len(name) > 16:
            warnings.warn('GenBank LOCUS line identifier over 16 characters', BiopythonParserWarning)
        consumer.locus(name)
        consumer.size(length)
        if line[33:51].strip() == '' and line[29:33] == ' aa ':
            consumer.residue_type('PROTEIN')
        else:
            consumer.residue_type(line[33:51].strip())
        consumer.molecule_type(line[33:41].strip())
        consumer.topology(line[42:51].strip())
        consumer.data_file_division(line[52:55])
        if line[62:73].strip():
            consumer.date(line[62:73])
    elif line[40:44] in [' bp ', ' aa ', ' rc '] and line[54:64].strip() in ['', 'linear', 'circular']:
        if len(line) < 79:
            warnings.warn(f'Truncated LOCUS line found - is this correct?\n:{line!r}', BiopythonParserWarning)
            padding_len = 79 - len(line)
            padding = ' ' * padding_len
            line += padding
        if line[40:44] not in [' bp ', ' aa ', ' rc ']:
            raise ValueError('LOCUS line does not contain size units at expected position:\n' + line)
        if line[44:47] not in ['   ', 'ss-', 'ds-', 'ms-']:
            raise ValueError('LOCUS line does not have valid strand type (Single stranded, ...):\n' + line)
        if not (line[47:54].strip() == '' or 'DNA' in line[47:54].strip().upper() or 'RNA' in line[47:54].strip().upper()):
            raise ValueError('LOCUS line does not contain valid sequence type (DNA, RNA, ...):\n' + line)
        if line[54:55] != ' ':
            raise ValueError('LOCUS line does not contain space at position 55:\n' + line)
        if line[55:63].strip() not in ['', 'linear', 'circular']:
            raise ValueError('LOCUS line does not contain valid entry (linear, circular, ...):\n' + line)
        if line[63:64] != ' ':
            raise ValueError('LOCUS line does not contain space at position 64:\n' + line)
        if line[67:68] != ' ':
            raise ValueError('LOCUS line does not contain space at position 68:\n' + line)
        if line[68:79].strip():
            if line[70:71] != '-':
                raise ValueError('LOCUS line does not contain - at position 71 in date:\n' + line)
            if line[74:75] != '-':
                raise ValueError('LOCUS line does not contain - at position 75 in date:\n' + line)
        name_and_length_str = line[self.GENBANK_INDENT:40]
        while '  ' in name_and_length_str:
            name_and_length_str = name_and_length_str.replace('  ', ' ')
        name_and_length = name_and_length_str.split(' ')
        if len(name_and_length) > 2:
            raise ValueError('Cannot parse the name and length in the LOCUS line:\n' + line)
        if len(name_and_length) == 1:
            raise ValueError('Name and length collide in the LOCUS line:\n' + line)
        consumer.locus(name_and_length[0])
        consumer.size(name_and_length[1])
        if line[44:54].strip() == '' and line[40:44] == ' aa ':
            consumer.residue_type(('PROTEIN ' + line[54:63]).strip())
        else:
            consumer.residue_type(line[44:63].strip())
        consumer.molecule_type(line[44:54].strip())
        consumer.topology(line[55:63].strip())
        if line[64:76].strip():
            consumer.data_file_division(line[64:67])
        if line[68:79].strip():
            consumer.date(line[68:79])
    elif line[self.GENBANK_INDENT:].strip().count(' ') == 0:
        if line[self.GENBANK_INDENT:].strip() != '':
            consumer.locus(line[self.GENBANK_INDENT:].strip())
        else:
            warnings.warn(f'Minimal LOCUS line found - is this correct?\n:{line!r}', BiopythonParserWarning)
    elif len(line.split()) == 8 and line.split()[3] in ('aa', 'bp') and (line.split()[5] in ('linear', 'circular')):
        splitline = line.split()
        consumer.locus(splitline[1])
        if int(splitline[2]) > sys.maxsize:
            raise ValueError('Tried to load a sequence with a length %s, your installation of python can only load sesquences of length %s' % (splitline[2], sys.maxsize))
        else:
            consumer.size(splitline[2])
        consumer.residue_type(splitline[4])
        consumer.topology(splitline[5])
        consumer.data_file_division(splitline[6])
        consumer.date(splitline[7])
        if len(line) < 80:
            warnings.warn('Attempting to parse malformed locus line:\n%r\nFound locus %r size %r residue_type %r\nSome fields may be wrong.' % (line, splitline[1], splitline[2], splitline[4]), BiopythonParserWarning)
    elif len(line.split()) == 7 and line.split()[3] in ['aa', 'bp']:
        splitline = line.split()
        consumer.locus(splitline[1])
        consumer.size(splitline[2])
        consumer.residue_type(splitline[4])
        consumer.data_file_division(splitline[5])
        consumer.date(splitline[6])
    elif len(line.split()) >= 4 and line.split()[3] in ['aa', 'bp']:
        warnings.warn(f'Malformed LOCUS line found - is this correct?\n:{line!r}', BiopythonParserWarning)
        consumer.locus(line.split()[1])
        consumer.size(line.split()[2])
    elif len(line.split()) >= 4 and line.split()[-1] in ['aa', 'bp']:
        warnings.warn(f'Malformed LOCUS line found - is this correct?\n:{line!r}', BiopythonParserWarning)
        consumer.locus(line[5:].rsplit(None, 2)[0].strip())
        consumer.size(line.split()[-2])
    else:
        raise ValueError('Did not recognise the LOCUS line layout:\n' + line)