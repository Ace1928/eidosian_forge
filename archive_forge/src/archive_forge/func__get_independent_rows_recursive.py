from ..pari import pari
import fractions
def _get_independent_rows_recursive(row_explain_pairs, length, desired_determinant, selected_rows, selected_explains):
    if len(selected_rows) == length:
        if desired_determinant is None:
            return selected_explains
        determinant = _internal_to_pari(selected_rows).matdet().abs()
        if determinant == desired_determinant:
            return selected_explains
        else:
            return None
    for row, explain in row_explain_pairs:
        new_selected_rows = selected_rows + [row]
        new_selected_explains = selected_explains + [explain]
        if has_full_rank(new_selected_rows):
            result = _get_independent_rows_recursive(row_explain_pairs, length, desired_determinant, new_selected_rows, new_selected_explains)
            if result:
                return result
    return None