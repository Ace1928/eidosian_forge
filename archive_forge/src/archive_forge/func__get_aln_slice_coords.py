import re
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _get_aln_slice_coords(parsed_hsp):
    """Get HSPs sequences (PRIVATE).

    To get the actual pairwise alignment sequences, we must first
    translate the un-gapped sequence based coordinates into positions
    in the gapped sequence (which may have a flanking region shown
    using leading - characters).  To date, I have never seen any
    trailing flanking region shown in the m10 file, but the
    following code should also cope with that.

    Note that this code seems to work fine even when the "sq_offset"
    entries are present as a result of using the -X command line option.
    """
    seq = parsed_hsp['seq']
    seq_stripped = seq.strip('-')
    disp_start = int(parsed_hsp['_display_start'])
    start = int(parsed_hsp['_start'])
    stop = int(parsed_hsp['_stop'])
    if start <= stop:
        start = start - disp_start
        stop = stop - disp_start + 1
    else:
        start = disp_start - start
        stop = disp_start - stop + 1
    stop += seq_stripped.count('-')
    if not (0 <= start and start < stop and (stop <= len(seq_stripped))):
        raise ValueError('Problem with sequence start/stop,\n%s[%i:%i]\n%s' % (seq, start, stop, parsed_hsp))
    return (start, stop)