def _ghd_aux(mat, rowv, colv, ins_cost, del_cost, shift_cost_coeff):
    for i, rowi in enumerate(rowv):
        for j, colj in enumerate(colv):
            shift_cost = shift_cost_coeff * abs(rowi - colj) + mat[i, j]
            if rowi == colj:
                tcost = mat[i, j]
            elif rowi > colj:
                tcost = del_cost + mat[i, j + 1]
            else:
                tcost = ins_cost + mat[i + 1, j]
            mat[i + 1, j + 1] = min(tcost, shift_cost)