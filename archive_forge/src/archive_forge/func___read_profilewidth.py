import re
def __read_profilewidth(record, line):
    if 'Nseqs1' not in line:
        raise ValueError(f"Line does not contain 'Nseqs1':\n{line}")
    m = __regex['profilewidth'].search(line)
    record.query_nseqs = int(m.group(1))
    record.query_neffseqs = float(m.group(2))
    record.hit_nseqs = int(m.group(3))
    record.hit_neffseqs = float(m.group(4))