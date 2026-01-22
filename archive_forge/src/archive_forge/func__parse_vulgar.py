import numpy as np
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
@staticmethod
def _parse_vulgar(words):
    query_id = words[0]
    query_start = int(words[1])
    query_end = int(words[2])
    query_strand = words[3]
    target_id = words[4]
    target_start = int(words[5])
    target_end = int(words[6])
    target_strand = words[7]
    score = float(words[8])
    ops = words[9::3]
    qs = 0
    ts = 0
    n = (len(words) - 8) // 3 + ops.count('N')
    coordinates = np.empty((2, n + 1), int)
    coordinates[0, 0] = ts
    coordinates[1, 0] = qs
    operations = bytearray(n)
    i = 0
    for operation, query_step, target_step in zip(ops, words[10::3], words[11::3]):
        query_step = int(query_step)
        target_step = int(target_step)
        if operation == 'M':
            pass
        elif operation == '5':
            assert target_step == 2 or query_step == 2
        elif operation == 'I':
            operation = 'N'
        elif operation == '3':
            assert target_step == 2 or query_step == 2
        elif operation == 'C':
            assert target_step % 3 == 0
            assert query_step % 3 == 0
        elif operation == 'G':
            if query_step == 0:
                operation = 'D'
            elif target_step == 0:
                operation = 'I'
            else:
                raise ValueError('Unexpected gap operation with steps %d, %d in vulgar line' % (query_step, target_step))
        elif operation == 'N':
            operation = 'U'
            if target_step > 0:
                ts += target_step
                coordinates[0, i + 1] = ts
                coordinates[1, i + 1] = qs
                operations[i] = ord(operation)
                i += 1
            if query_step > 0:
                qs += query_step
                coordinates[0, i + 1] = ts
                coordinates[1, i + 1] = qs
                operations[i] = ord(operation)
                i += 1
            continue
        elif operation == 'S':
            pass
        elif operation == 'F':
            pass
        else:
            raise ValueError('Unknown operation %s in vulgar string' % operation)
        ts += target_step
        qs += query_step
        coordinates[0, i + 1] = ts
        coordinates[1, i + 1] = qs
        operations[i] = ord(operation)
        i += 1
    if target_strand == '+':
        coordinates[0, :] += target_start
        target_length = target_end
        target_molecule_type = None
    elif target_strand == '-':
        coordinates[0, :] = target_start - coordinates[0, :]
        target_length = target_start
        target_molecule_type = None
    elif target_strand == '.':
        coordinates[0, :] += target_start
        target_molecule_type = 'protein'
        target_length = target_end
    if query_strand == '+':
        coordinates[1, :] += query_start
        query_length = query_end
        query_molecule_type = None
    elif query_strand == '-':
        coordinates[1, :] = query_start - coordinates[1, :]
        query_length = query_start
        query_molecule_type = None
    elif query_strand == '.':
        coordinates[1, :] += query_start
        query_molecule_type = 'protein'
        query_length = query_end
    target_seq = Seq(None, length=target_length)
    query_seq = Seq(None, length=query_length)
    target = SeqRecord(target_seq, id=target_id, description='')
    query = SeqRecord(query_seq, id=query_id, description='')
    if target_molecule_type is not None:
        target.annotations['molecule_type'] = target_molecule_type
    if query_molecule_type is not None:
        query.annotations['molecule_type'] = query_molecule_type
    alignment = Alignment([target, query], coordinates)
    alignment.operations = operations
    alignment.score = score
    return alignment