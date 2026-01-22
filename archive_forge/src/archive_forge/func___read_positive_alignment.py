import re
def __read_positive_alignment(record, line):
    m = __regex['positive_alignment'].match(line)
    assert m is not None, 'invalid match'
    record.positives += m.group(1)