import re
def __read_scores(record, line):
    if not line.startswith('Smith-Waterman'):
        raise ValueError(f"Line does not start with 'Smith-Waterman':\n{line}")
    m = __regex['scores'].search(line)
    if m:
        record.sw_score = int(m.group(1))
        record.evalue = float(m.group(2))
    else:
        record.sw_score = 0
        record.evalue = -1.0