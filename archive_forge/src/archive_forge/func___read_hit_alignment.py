import re
def __read_hit_alignment(record, line):
    m = __regex['start'].search(line)
    if m:
        record.hit_start = int(m.group(1))
    m = __regex['align'].match(line)
    assert m is not None, 'invalid match'
    record.hit_aln += m.group(1)