import warnings
import re
import sys
from collections import defaultdict
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonParserWarning
from typing import List
class GenBankScanner(InsdcScanner):
    """For extracting chunks of information in GenBank files."""
    RECORD_START = 'LOCUS       '
    HEADER_WIDTH = 12
    FEATURE_START_MARKERS = ['FEATURES             Location/Qualifiers', 'FEATURES']
    FEATURE_END_MARKERS: List[str] = []
    FEATURE_QUALIFIER_INDENT = 21
    FEATURE_QUALIFIER_SPACER = ' ' * FEATURE_QUALIFIER_INDENT
    SEQUENCE_HEADERS = ['CONTIG', 'ORIGIN', 'BASE COUNT', 'WGS', 'TSA', 'TLS']
    GENBANK_INDENT = HEADER_WIDTH
    GENBANK_SPACER = ' ' * GENBANK_INDENT
    STRUCTURED_COMMENT_START = '-START##'
    STRUCTURED_COMMENT_END = '-END##'
    STRUCTURED_COMMENT_DELIM = ' :: '

    def parse_footer(self):
        """Return a tuple containing a list of any misc strings, and the sequence."""
        if self.line[:self.HEADER_WIDTH].rstrip() not in self.SEQUENCE_HEADERS:
            raise ValueError(f"Footer format unexpected:  '{self.line}'")
        misc_lines = []
        while self.line[:self.HEADER_WIDTH].rstrip() in self.SEQUENCE_HEADERS or self.line[:self.HEADER_WIDTH] == ' ' * self.HEADER_WIDTH or 'WGS' == self.line[:3]:
            misc_lines.append(self.line.rstrip())
            self.line = self.handle.readline()
            if not self.line:
                raise ValueError('Premature end of file')
        if self.line[:self.HEADER_WIDTH].rstrip() in self.SEQUENCE_HEADERS:
            raise ValueError(f"Eh? '{self.line}'")
        seq_lines = []
        line = self.line
        while True:
            if not line:
                warnings.warn('Premature end of file in sequence data', BiopythonParserWarning)
                line = '//'
                break
            line = line.rstrip()
            if not line:
                warnings.warn('Blank line in sequence data', BiopythonParserWarning)
                line = self.handle.readline()
                continue
            if line == '//':
                break
            if line.startswith('CONTIG'):
                break
            if len(line) > 9 and line[9:10] != ' ':
                warnings.warn('Invalid indentation for sequence line', BiopythonParserWarning)
                line = line[1:]
                if len(line) > 9 and line[9:10] != ' ':
                    raise ValueError(f"Sequence line mal-formed, '{line}'")
            seq_lines.append(line[10:])
            line = self.handle.readline()
        self.line = line
        return (misc_lines, ''.join(seq_lines).replace(' ', ''))

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

    def _feed_header_lines(self, consumer, lines):
        consumer_dict = {'DEFINITION': 'definition', 'ACCESSION': 'accession', 'NID': 'nid', 'PID': 'pid', 'DBSOURCE': 'db_source', 'KEYWORDS': 'keywords', 'SEGMENT': 'segment', 'SOURCE': 'source', 'AUTHORS': 'authors', 'CONSRTM': 'consrtm', 'PROJECT': 'project', 'TITLE': 'title', 'JOURNAL': 'journal', 'MEDLINE': 'medline_id', 'PUBMED': 'pubmed_id', 'REMARK': 'remark'}
        lines = [_f for _f in lines if _f]
        lines.append('')
        line_iter = iter(lines)
        try:
            line = next(line_iter)
            while True:
                if not line:
                    break
                line_type = line[:self.GENBANK_INDENT].strip()
                data = line[self.GENBANK_INDENT:].strip()
                if line_type == 'VERSION':
                    while '  ' in data:
                        data = data.replace('  ', ' ')
                    if ' GI:' not in data:
                        consumer.version(data)
                    else:
                        if self.debug:
                            print('Version [' + data.split(' GI:')[0] + '], gi [' + data.split(' GI:')[1] + ']')
                        consumer.version(data.split(' GI:')[0])
                        consumer.gi(data.split(' GI:')[1])
                    line = next(line_iter)
                elif line_type == 'DBLINK':
                    consumer.dblink(data.strip())
                    while True:
                        line = next(line_iter)
                        if line[:self.GENBANK_INDENT] == self.GENBANK_SPACER:
                            consumer.dblink(line[self.GENBANK_INDENT:].strip())
                        else:
                            break
                elif line_type == 'REFERENCE':
                    if self.debug > 1:
                        print('Found reference [' + data + ']')
                    data = data.strip()
                    while True:
                        line = next(line_iter)
                        if line[:self.GENBANK_INDENT] == self.GENBANK_SPACER:
                            data += ' ' + line[self.GENBANK_INDENT:]
                            if self.debug > 1:
                                print('Extended reference text [' + data + ']')
                        else:
                            break
                    while '  ' in data:
                        data = data.replace('  ', ' ')
                    if ' ' not in data:
                        if self.debug > 2:
                            print('Reference number "' + data + '"')
                        consumer.reference_num(data)
                    else:
                        if self.debug > 2:
                            print('Reference number "' + data[:data.find(' ')] + '", "' + data[data.find(' ') + 1:] + '"')
                        consumer.reference_num(data[:data.find(' ')])
                        consumer.reference_bases(data[data.find(' ') + 1:])
                elif line_type == 'ORGANISM':
                    organism_data = data
                    lineage_data = ''
                    while True:
                        line = next(line_iter)
                        if line[0:self.GENBANK_INDENT] == self.GENBANK_SPACER:
                            if lineage_data or ';' in line or line[self.GENBANK_INDENT:].strip() in ('Bacteria.', 'Archaea.', 'Eukaryota.', 'Unclassified.', 'Viruses.', 'cellular organisms.', 'other sequences.', 'unclassified sequences.'):
                                lineage_data += ' ' + line[self.GENBANK_INDENT:]
                            elif line[self.GENBANK_INDENT:].strip() == '.':
                                pass
                            else:
                                organism_data += ' ' + line[self.GENBANK_INDENT:].strip()
                        else:
                            break
                    consumer.organism(organism_data)
                    if lineage_data.strip() == '' and self.debug > 1:
                        print('Taxonomy line(s) missing or blank')
                    consumer.taxonomy(lineage_data.strip())
                    del organism_data, lineage_data
                elif line_type == 'COMMENT':
                    data = line[self.GENBANK_INDENT:]
                    if self.debug > 1:
                        print('Found comment')
                    comment_list = []
                    structured_comment_dict = defaultdict(dict)
                    regex = f'([^#]+){self.STRUCTURED_COMMENT_START}$'
                    structured_comment_key = re.search(regex, data)
                    if structured_comment_key is not None:
                        structured_comment_key = structured_comment_key.group(1)
                        if self.debug > 1:
                            print('Found Structured Comment')
                    else:
                        comment_list.append(data)
                    while True:
                        line = next(line_iter)
                        data = line[self.GENBANK_INDENT:]
                        if line[0:self.GENBANK_INDENT] == self.GENBANK_SPACER:
                            if self.STRUCTURED_COMMENT_START in data:
                                regex = f'([^#]+){self.STRUCTURED_COMMENT_START}$'
                                structured_comment_key = re.search(regex, data)
                                if structured_comment_key is not None:
                                    structured_comment_key = structured_comment_key.group(1)
                                else:
                                    comment_list.append(data)
                            elif structured_comment_key is not None and self.STRUCTURED_COMMENT_DELIM.strip() in data:
                                match = re.search(f'(.+?)\\s*{self.STRUCTURED_COMMENT_DELIM.strip()}\\s*(.*)', data)
                                structured_comment_dict[structured_comment_key][match.group(1)] = match.group(2)
                                if self.debug > 2:
                                    print('Structured Comment continuation [' + data + ']')
                            elif structured_comment_key is not None and self.STRUCTURED_COMMENT_END not in data:
                                if structured_comment_key not in structured_comment_dict:
                                    warnings.warn('Structured comment not parsed for %s. Is it malformed?' % consumer.data.name, BiopythonParserWarning)
                                    continue
                                previous_value_line = structured_comment_dict[structured_comment_key][match.group(1)]
                                structured_comment_dict[structured_comment_key][match.group(1)] = previous_value_line + ' ' + line.strip()
                            elif self.STRUCTURED_COMMENT_END in data:
                                structured_comment_key = None
                            else:
                                comment_list.append(data)
                                if self.debug > 2:
                                    print('Comment continuation [' + data + ']')
                        else:
                            break
                    if comment_list:
                        consumer.comment(comment_list)
                    if structured_comment_dict:
                        consumer.structured_comment(structured_comment_dict)
                    del comment_list, structured_comment_key, structured_comment_dict
                elif line_type in consumer_dict:
                    while True:
                        line = next(line_iter)
                        if line[0:self.GENBANK_INDENT] == self.GENBANK_SPACER:
                            data += ' ' + line[self.GENBANK_INDENT:]
                        else:
                            if line_type == 'DEFINITION' and data.endswith('.'):
                                data = data[:-1]
                            getattr(consumer, consumer_dict[line_type])(data)
                            break
                else:
                    if self.debug:
                        print('Ignoring GenBank header line:\n' % line)
                    line = next(line_iter)
        except StopIteration:
            raise ValueError('Problem in header') from None

    def _feed_misc_lines(self, consumer, lines):
        lines.append('')
        line_iter = iter(lines)
        try:
            for line in line_iter:
                if line.startswith('BASE COUNT'):
                    line = line[10:].strip()
                    if line:
                        if self.debug:
                            print('base_count = ' + line)
                        consumer.base_count(line)
                if line.startswith('ORIGIN'):
                    line = line[6:].strip()
                    if line:
                        if self.debug:
                            print('origin_name = ' + line)
                        consumer.origin_name(line)
                if line.startswith('TLS '):
                    line = line[3:].strip()
                    consumer.tls(line)
                if line.startswith('TSA '):
                    line = line[3:].strip()
                    consumer.tsa(line)
                if line.startswith('WGS '):
                    line = line[3:].strip()
                    consumer.wgs(line)
                if line.startswith('WGS_SCAFLD'):
                    line = line[10:].strip()
                    consumer.add_wgs_scafld(line)
                if line.startswith('CONTIG'):
                    line = line[6:].strip()
                    contig_location = line
                    while True:
                        line = next(line_iter)
                        if not line:
                            break
                        elif line[:self.GENBANK_INDENT] == self.GENBANK_SPACER:
                            contig_location += line[self.GENBANK_INDENT:].rstrip()
                        elif line.startswith('ORIGIN'):
                            line = line[6:].strip()
                            if line:
                                consumer.origin_name(line)
                            break
                        else:
                            raise ValueError('Expected CONTIG continuation line, got:\n' + line)
                    consumer.contig_location(contig_location)
            return
        except StopIteration:
            raise ValueError('Problem in misc lines before sequence') from None