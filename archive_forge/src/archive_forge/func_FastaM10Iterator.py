from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def FastaM10Iterator(handle, seq_count=None):
    """Alignment iterator for the FASTA tool's pairwise alignment output.

    This is for reading the pairwise alignments output by Bill Pearson's
    FASTA program when called with the -m 10 command line option for machine
    readable output.  For more details about the FASTA tools, see the website
    http://fasta.bioch.virginia.edu/ and the paper:

         W.R. Pearson & D.J. Lipman PNAS (1988) 85:2444-2448

    This class is intended to be used via the Bio.AlignIO.parse() function
    by specifying the format as "fasta-m10" as shown in the following code::

        from Bio import AlignIO
        handle = ...
        for a in AlignIO.parse(handle, "fasta-m10"):
            assert len(a) == 2, "Should be pairwise!"
            print("Alignment length %i" % a.get_alignment_length())
            for record in a:
                print("%s %s %s" % (record.seq, record.name, record.id))

    Note that this is not a full blown parser for all the information
    in the FASTA output - for example, most of the header and all of the
    footer is ignored.  Also, the alignments are not batched according to
    the input queries.

    Also note that there can be up to about 30 letters of flanking region
    included in the raw FASTA output as contextual information.  This is NOT
    part of the alignment itself, and is not included in the resulting
    MultipleSeqAlignment objects returned.
    """
    state_PREAMBLE = -1
    state_NONE = 0
    state_QUERY_HEADER = 1
    state_ALIGN_HEADER = 2
    state_ALIGN_QUERY = 3
    state_ALIGN_MATCH = 4
    state_ALIGN_CONS = 5

    def build_hsp():
        if not query_tags and (not match_tags):
            raise ValueError(f'No data for query {query_id!r}, match {match_id!r}')
        assert query_tags, query_tags
        assert match_tags, match_tags
        evalue = align_tags.get('fa_expect')
        tool = global_tags.get('tool', '').upper()
        q = _extract_alignment_region(query_seq, query_tags)
        if tool in ['TFASTX'] and len(match_seq) == len(q):
            m = match_seq
        else:
            m = _extract_alignment_region(match_seq, match_tags)
        if len(q) != len(m):
            raise ValueError(f'Darn... amino acids vs nucleotide coordinates?\ntool: {tool}\nquery_seq: {query_seq}\nquery_tags: {query_tags}\n{q} length: {len(q)}\nmatch_seq: {match_seq}\nmatch_tags: {match_tags}\n{m} length: {len(m)}\nhandle.name: {handle.name}\n')
        annotations = {}
        records = []
        annotations.update(header_tags)
        annotations.update(align_tags)
        record = SeqRecord(Seq(q), id=query_id, name='query', description=query_descr, annotations={'original_length': int(query_tags['sq_len'])})
        record._al_start = int(query_tags['al_start'])
        record._al_stop = int(query_tags['al_stop'])
        if 'sq_type' in query_tags:
            if query_tags['sq_type'] == 'D':
                record.annotations['molecule_type'] = 'DNA'
            elif query_tags['sq_type'] == 'p':
                record.annotations['molecule_type'] = 'protein'
        records.append(record)
        record = SeqRecord(Seq(m), id=match_id, name='match', description=match_descr, annotations={'original_length': int(match_tags['sq_len'])})
        record._al_start = int(match_tags['al_start'])
        record._al_stop = int(match_tags['al_stop'])
        if 'sq_type' in match_tags:
            if match_tags['sq_type'] == 'D':
                record.annotations['molecule_type'] = 'DNA'
            elif match_tags['sq_type'] == 'p':
                record.annotations['molecule_type'] = 'protein'
        records.append(record)
        return MultipleSeqAlignment(records, annotations=annotations)
    state = state_PREAMBLE
    query_id = None
    match_id = None
    query_descr = ''
    match_descr = ''
    global_tags = {}
    header_tags = {}
    align_tags = {}
    query_tags = {}
    match_tags = {}
    query_seq = ''
    match_seq = ''
    cons_seq = ''
    for line in handle:
        if '>>>' in line and (not line.startswith('>>>')):
            if query_id and match_id:
                yield build_hsp()
            state = state_NONE
            query_descr = line[line.find('>>>') + 3:].strip()
            query_id = query_descr.split(None, 1)[0]
            match_id = None
            header_tags = {}
            align_tags = {}
            query_tags = {}
            match_tags = {}
            query_seq = ''
            match_seq = ''
            cons_seq = ''
        elif line.startswith('!! No '):
            assert state == state_NONE
            assert not header_tags
            assert not align_tags
            assert not match_tags
            assert not query_tags
            assert match_id is None
            assert not query_seq
            assert not match_seq
            assert not cons_seq
            query_id = None
        elif line.strip() in ['>>><<<', '>>>///']:
            if query_id and match_id:
                yield build_hsp()
            state = state_NONE
            query_id = None
            match_id = None
            header_tags = {}
            align_tags = {}
            query_tags = {}
            match_tags = {}
            query_seq = ''
            match_seq = ''
            cons_seq = ''
        elif line.startswith('>>>'):
            assert query_id is not None
            assert line[3:].split(', ', 1)[0] == query_id, line
            assert match_id is None
            assert not header_tags
            assert not align_tags
            assert not query_tags
            assert not match_tags
            assert not match_seq
            assert not query_seq
            assert not cons_seq
            state = state_QUERY_HEADER
        elif line.startswith('>>'):
            if query_id and match_id:
                yield build_hsp()
            align_tags = {}
            query_tags = {}
            match_tags = {}
            query_seq = ''
            match_seq = ''
            cons_seq = ''
            match_descr = line[2:].strip()
            match_id = match_descr.split(None, 1)[0]
            state = state_ALIGN_HEADER
        elif line.startswith('>--'):
            assert query_id and match_id, line
            yield build_hsp()
            align_tags = {}
            query_tags = {}
            match_tags = {}
            query_seq = ''
            match_seq = ''
            cons_seq = ''
            state = state_ALIGN_HEADER
        elif line.startswith('>'):
            if state == state_ALIGN_HEADER:
                assert query_id is not None, line
                assert match_id is not None, line
                assert query_id.startswith(line[1:].split(None, 1)[0]), line
                state = state_ALIGN_QUERY
            elif state == state_ALIGN_QUERY:
                assert query_id is not None, line
                assert match_id is not None, line
                assert match_id.startswith(line[1:].split(None, 1)[0]), line
                state = state_ALIGN_MATCH
            elif state == state_NONE:
                pass
            else:
                raise RuntimeError('state %i got %r' % (state, line))
        elif line.startswith('; al_cons'):
            assert state == state_ALIGN_MATCH, line
            state = state_ALIGN_CONS
        elif line.startswith('; '):
            if ': ' in line:
                key, value = (s.strip() for s in line[2:].split(': ', 1))
            else:
                import warnings
                from Bio import BiopythonParserWarning
                warnings.warn(f'Missing colon in line: {line!r}', BiopythonParserWarning)
                try:
                    key, value = (s.strip() for s in line[2:].split(' ', 1))
                except ValueError:
                    raise ValueError(f'Bad line: {line!r}') from None
            if state == state_QUERY_HEADER:
                header_tags[key] = value
            elif state == state_ALIGN_HEADER:
                align_tags[key] = value
            elif state == state_ALIGN_QUERY:
                query_tags[key] = value
            elif state == state_ALIGN_MATCH:
                match_tags[key] = value
            else:
                raise RuntimeError(f'Unexpected state {state!r}, {line!r}')
        elif state == state_ALIGN_QUERY:
            query_seq += line.strip()
        elif state == state_ALIGN_MATCH:
            match_seq += line.strip()
        elif state == state_ALIGN_CONS:
            cons_seq += line.strip('\n')
        elif state == state_PREAMBLE:
            if line.startswith('#'):
                global_tags['command'] = line[1:].strip()
            elif line.startswith(' version '):
                global_tags['version'] = line[9:].strip()
            elif ' compares a ' in line:
                global_tags['tool'] = line[:line.find(' compares a ')].strip()
            elif ' searches a ' in line:
                global_tags['tool'] = line[:line.find(' searches a ')].strip()
        else:
            pass