from ..pari import pari
import fractions
def _expand_square_matrix(m, num_cols_rows):

    def upleft(row):
        return m[row]

    def upright(row):
        return [0 for col in range(num_cols_rows)]

    def downleft(row):
        return [0 for col in range(len(m))]

    def downright(row):
        return [1 if row == col else 0 for col in range(num_cols_rows)]
    up = [upleft(row) + upright(row) for row in range(len(m))]
    down = [downleft(row) + downright(row) for row in range(num_cols_rows)]
    return up + down