import warnings
from collections import namedtuple
from Bio import BiopythonWarning
from Bio import BiopythonDeprecationWarning
from Bio.Align import substitution_matrices
def _make_score_matrix_fast(sequenceA, sequenceB, match_fn, open_A, extend_A, open_B, extend_B, penalize_extend_when_opening, penalize_end_gaps, align_globally, score_only):
    """Generate a score and traceback matrix according to Gotoh (PRIVATE).

    This is an implementation of the Needleman-Wunsch dynamic programming
    algorithm as modified by Gotoh, implementing affine gap penalties.
    In short, we have three matrices, holding scores for alignments ending
    in (1) a match/mismatch, (2) a gap in sequence A, and (3) a gap in
    sequence B, respectively. However, we can combine them in one matrix,
    which holds the best scores, and store only those values from the
    other matrices that are actually used for the next step of calculation.
    The traceback matrix holds the positions for backtracing the alignment.
    """
    first_A_gap = calc_affine_penalty(1, open_A, extend_A, penalize_extend_when_opening)
    first_B_gap = calc_affine_penalty(1, open_B, extend_B, penalize_extend_when_opening)
    local_max_score = 0
    lenA, lenB = (len(sequenceA), len(sequenceB))
    score_matrix, trace_matrix = ([], [])
    for i in range(lenA + 1):
        score_matrix.append([None] * (lenB + 1))
        if not score_only:
            trace_matrix.append([None] * (lenB + 1))
    for i in range(lenA + 1):
        if penalize_end_gaps[1]:
            score = calc_affine_penalty(i, open_B, extend_B, penalize_extend_when_opening)
        else:
            score = 0
        score_matrix[i][0] = score
    for i in range(lenB + 1):
        if penalize_end_gaps[0]:
            score = calc_affine_penalty(i, open_A, extend_A, penalize_extend_when_opening)
        else:
            score = 0
        score_matrix[0][i] = score
    col_score = [0]
    for i in range(1, lenB + 1):
        col_score.append(calc_affine_penalty(i, 2 * open_B, extend_B, penalize_extend_when_opening))
    for row in range(1, lenA + 1):
        row_score = calc_affine_penalty(row, 2 * open_A, extend_A, penalize_extend_when_opening)
        for col in range(1, lenB + 1):
            nogap_score = score_matrix[row - 1][col - 1] + match_fn(sequenceA[row - 1], sequenceB[col - 1])
            if not penalize_end_gaps[0] and row == lenA:
                row_open = score_matrix[row][col - 1]
                row_extend = row_score
            else:
                row_open = score_matrix[row][col - 1] + first_A_gap
                row_extend = row_score + extend_A
            row_score = max(row_open, row_extend)
            if not penalize_end_gaps[1] and col == lenB:
                col_open = score_matrix[row - 1][col]
                col_extend = col_score[col]
            else:
                col_open = score_matrix[row - 1][col] + first_B_gap
                col_extend = col_score[col] + extend_B
            col_score[col] = max(col_open, col_extend)
            best_score = max(nogap_score, col_score[col], row_score)
            local_max_score = max(local_max_score, best_score)
            if not align_globally and best_score < 0:
                score_matrix[row][col] = 0
            else:
                score_matrix[row][col] = best_score
            if not score_only:
                row_score_rint = rint(row_score)
                col_score_rint = rint(col_score[col])
                row_trace_score = 0
                col_trace_score = 0
                if rint(row_open) == row_score_rint:
                    row_trace_score += 1
                if rint(row_extend) == row_score_rint:
                    row_trace_score += 8
                if rint(col_open) == col_score_rint:
                    col_trace_score += 4
                if rint(col_extend) == col_score_rint:
                    col_trace_score += 16
                trace_score = 0
                best_score_rint = rint(best_score)
                if rint(nogap_score) == best_score_rint:
                    trace_score += 2
                if row_score_rint == best_score_rint:
                    trace_score += row_trace_score
                if col_score_rint == best_score_rint:
                    trace_score += col_trace_score
                trace_matrix[row][col] = trace_score
    if not align_globally:
        best_score = local_max_score
    return (score_matrix, trace_matrix, best_score)