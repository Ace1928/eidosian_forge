from ..pari import pari
import fractions
def _bottom_row_stable_smith_normal_form(m):
    m_up, m_down = _split_matrix_bottom_zero_rows(m)
    if len(m_up) == 0:
        return (_identity_matrix(len(m)), _identity_matrix(len(m[0])), m)
    u_upleft, v, d_up = _smith_normal_form_with_inverse(m_up)
    return (_expand_square_matrix(u_upleft, len(m_down)), v, d_up + m_down)