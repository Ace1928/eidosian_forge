import re
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _set_hsp_seqs(hsp, parsed, program):
    """Set HSPs sequences (PRIVATE).

    :param hsp: HSP whose properties will be set
    :type hsp: HSP
    :param parsed: parsed values of the HSP attributes
    :type parsed: dictionary {string: object}
    :param program: program name
    :type program: string

    """
    start = 0
    for seq_type in ('hit', 'query'):
        if 'tfast' not in program:
            pseq = parsed[seq_type]
            start, stop = _get_aln_slice_coords(pseq)
            start_adj = len(re.search(_RE_START_EXC, pseq['seq']).group(0))
            stop_adj = len(re.search(_RE_END_EXC, pseq['seq']).group(0))
            start = start + start_adj
            stop = stop + start_adj - stop_adj
            parsed[seq_type]['seq'] = pseq['seq'][start:stop]
    if len(parsed['query']['seq']) != len(parsed['hit']['seq']):
        raise ValueError('Length mismatch: %r %r' % (len(parsed['query']['seq']), len(parsed['hit']['seq'])))
    if 'similarity' in hsp.aln_annotation:
        hsp.aln_annotation['similarity'] = hsp.aln_annotation['similarity'][start:]
        assert len(hsp.aln_annotation['similarity']) == len(parsed['hit']['seq'])
    assert parsed['query']['_type'] == parsed['hit']['_type']
    type_val = parsed['query']['_type']
    molecule_type = 'DNA' if type_val == 'D' else 'protein'
    setattr(hsp.fragment, 'molecule_type', molecule_type)
    for seq_type in ('hit', 'query'):
        start = int(parsed[seq_type]['_start'])
        end = int(parsed[seq_type]['_stop'])
        setattr(hsp.fragment, seq_type + '_start', min(start, end) - 1)
        setattr(hsp.fragment, seq_type + '_end', max(start, end))
        setattr(hsp.fragment, seq_type, parsed[seq_type]['seq'])
        if molecule_type != 'protein':
            if start <= end:
                setattr(hsp.fragment, seq_type + '_strand', 1)
            else:
                setattr(hsp.fragment, seq_type + '_strand', -1)
        else:
            setattr(hsp.fragment, seq_type + '_strand', 0)