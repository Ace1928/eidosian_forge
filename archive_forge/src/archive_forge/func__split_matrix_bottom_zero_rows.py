from ..pari import pari
import fractions
def _split_matrix_bottom_zero_rows(m):
    for number_top_rows in range(len(m), -1, -1):
        if not row_is_zero(m, number_top_rows - 1):
            break
    return (m[:number_top_rows], m[number_top_rows:])