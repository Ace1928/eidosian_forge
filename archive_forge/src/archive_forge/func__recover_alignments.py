import warnings
from collections import namedtuple
from Bio import BiopythonWarning
from Bio import BiopythonDeprecationWarning
from Bio.Align import substitution_matrices
def _recover_alignments(sequenceA, sequenceB, starts, best_score, score_matrix, trace_matrix, align_globally, gap_char, one_alignment_only, gap_A_fn, gap_B_fn, reverse=False):
    """Do the backtracing and return a list of alignments (PRIVATE).

    Recover the alignments by following the traceback matrix.  This
    is a recursive procedure, but it's implemented here iteratively
    with a stack.

    sequenceA and sequenceB may be sequences, including strings,
    lists, or list-like objects.  In order to preserve the type of
    the object, we need to use slices on the sequences instead of
    indexes.  For example, sequenceA[row] may return a type that's
    not compatible with sequenceA, e.g. if sequenceA is a list and
    sequenceA[row] is a string.  Thus, avoid using indexes and use
    slices, e.g. sequenceA[row:row+1].  Assume that client-defined
    sequence classes preserve these semantics.
    """
    lenA, lenB = (len(sequenceA), len(sequenceB))
    ali_seqA, ali_seqB = (sequenceA[0:0], sequenceB[0:0])
    tracebacks = []
    in_process = []
    for start in starts:
        score, (row, col) = start
        begin = 0
        if align_globally:
            end = None
        else:
            if (score, (row - 1, col - 1)) in starts:
                continue
            if score <= 0:
                continue
            trace = trace_matrix[row][col]
            if (trace - trace % 2) % 4 == 2:
                trace_matrix[row][col] = 2
            else:
                continue
            end = -max(lenA - row, lenB - col)
            if not end:
                end = None
            col_distance = lenB - col
            row_distance = lenA - row
            ali_seqA = (col_distance - row_distance) * gap_char + sequenceA[lenA - 1:row - 1:-1]
            ali_seqB = (row_distance - col_distance) * gap_char + sequenceB[lenB - 1:col - 1:-1]
        in_process += [(ali_seqA, ali_seqB, end, row, col, False, trace_matrix[row][col])]
    while in_process and len(tracebacks) < MAX_ALIGNMENTS:
        dead_end = False
        ali_seqA, ali_seqB, end, row, col, col_gap, trace = in_process.pop()
        while (row > 0 or col > 0) and (not dead_end):
            cache = (ali_seqA[:], ali_seqB[:], end, row, col, col_gap)
            if not trace:
                if col and col_gap:
                    dead_end = True
                else:
                    ali_seqA, ali_seqB = _finish_backtrace(sequenceA, sequenceB, ali_seqA, ali_seqB, row, col, gap_char)
                break
            elif trace % 2 == 1:
                trace -= 1
                if col_gap:
                    dead_end = True
                else:
                    col -= 1
                    ali_seqA += gap_char
                    ali_seqB += sequenceB[col:col + 1]
                    col_gap = False
            elif trace % 4 == 2:
                trace -= 2
                row -= 1
                col -= 1
                ali_seqA += sequenceA[row:row + 1]
                ali_seqB += sequenceB[col:col + 1]
                col_gap = False
            elif trace % 8 == 4:
                trace -= 4
                row -= 1
                ali_seqA += sequenceA[row:row + 1]
                ali_seqB += gap_char
                col_gap = True
            elif trace in (8, 24):
                trace -= 8
                if col_gap:
                    dead_end = True
                else:
                    col_gap = False
                    x = _find_gap_open(sequenceA, sequenceB, ali_seqA, ali_seqB, end, row, col, col_gap, gap_char, score_matrix, trace_matrix, in_process, gap_A_fn, col, row, 'col', best_score, align_globally)
                    ali_seqA, ali_seqB, row, col, in_process, dead_end = x
            elif trace == 16:
                trace -= 16
                col_gap = True
                x = _find_gap_open(sequenceA, sequenceB, ali_seqA, ali_seqB, end, row, col, col_gap, gap_char, score_matrix, trace_matrix, in_process, gap_B_fn, row, col, 'row', best_score, align_globally)
                ali_seqA, ali_seqB, row, col, in_process, dead_end = x
            if trace:
                cache += (trace,)
                in_process.append(cache)
            trace = trace_matrix[row][col]
            if not align_globally:
                if score_matrix[row][col] == best_score:
                    dead_end = True
                elif score_matrix[row][col] <= 0:
                    begin = max(row, col)
                    trace = 0
        if not dead_end:
            if not reverse:
                tracebacks.append((ali_seqA[::-1], ali_seqB[::-1], score, begin, end))
            else:
                tracebacks.append((ali_seqB[::-1], ali_seqA[::-1], score, begin, end))
            if one_alignment_only:
                break
    return _clean_alignments(tracebacks)