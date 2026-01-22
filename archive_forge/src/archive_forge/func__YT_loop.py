import warnings
import copy
from math import sqrt
import cupy
from cupyx.scipy import linalg
from cupyx.scipy.interpolate import make_interp_spline
from cupyx.scipy.linalg import expm, block_diag
from cupyx.scipy.signal._lti_conversion import (
from cupyx.scipy.signal._iir_filter_conversions import (
from cupyx.scipy.signal._filter_design import (
def _YT_loop(ker_pole, transfer_matrix, poles, B, maxiter, rtol):
    """
    Algorithm "YT" Tits, Yang. Globally Convergent
    Algorithms for Robust Pole Assignment by State Feedback
    https://hdl.handle.net/1903/5598
    The poles P have to be sorted accordingly to section 6.2 page 20

    """
    nb_real = poles[cupy.isreal(poles)].shape[0]
    hnb = nb_real // 2
    if nb_real > 0:
        update_order = [[cupy.array(nb_real)], [cupy.array(1)]]
    else:
        update_order = [[], []]
    r_comp = cupy.arange(nb_real + 1, len(poles) + 1, 2)
    r_p = cupy.arange(1, hnb + nb_real % 2)
    update_order[0].extend(2 * r_p)
    update_order[1].extend(2 * r_p + 1)
    update_order[0].extend(r_comp)
    update_order[1].extend(r_comp + 1)
    r_p = cupy.arange(1, hnb + 1)
    update_order[0].extend(2 * r_p - 1)
    update_order[1].extend(2 * r_p)
    if hnb == 0 and cupy.isreal(poles[0]):
        update_order[0].append(cupy.array(1))
        update_order[1].append(cupy.array(1))
    update_order[0].extend(r_comp)
    update_order[1].extend(r_comp + 1)
    r_j = cupy.arange(2, hnb + nb_real % 2)
    for j in r_j:
        for i in range(1, hnb + 1):
            update_order[0].append(cupy.array(i))
            update_order[1].append(cupy.array(i + j))
    if hnb == 0 and cupy.isreal(poles[0]):
        update_order[0].append(cupy.array(1))
        update_order[1].append(cupy.array(1))
    update_order[0].extend(r_comp)
    update_order[1].extend(r_comp + 1)
    r_j = cupy.arange(2, hnb + nb_real % 2)
    for j in r_j:
        for i in range(hnb + 1, nb_real + 1):
            idx_1 = i + j
            if idx_1 > nb_real:
                idx_1 = i + j - nb_real
            update_order[0].append(cupy.array(i))
            update_order[1].append(cupy.array(idx_1))
    if hnb == 0 and cupy.isreal(poles[0]):
        update_order[0].append(cupy.array(1))
        update_order[1].append(cupy.array(1))
    update_order[0].extend(r_comp)
    update_order[1].extend(r_comp + 1)
    for i in range(1, hnb + 1):
        update_order[0].append(cupy.array(i))
        update_order[1].append(cupy.array(i + hnb))
    if hnb == 0 and cupy.isreal(poles[0]):
        update_order[0].append(cupy.array(1))
        update_order[1].append(cupy.array(1))
    update_order[0].extend(r_comp)
    update_order[1].extend(r_comp + 1)
    update_order = cupy.array(update_order).T - 1
    stop = False
    nb_try = 0
    while nb_try < maxiter and (not stop):
        det_transfer_matrixb = cupy.abs(cupy.linalg.det(transfer_matrix))
        for i, j in update_order:
            i, j = (int(i), int(j))
            if i == j:
                assert i == 0, 'i!=0 for KNV call in YT'
                assert cupy.isreal(poles[i]), 'calling KNV on a complex pole'
                _KNV0(B, ker_pole, transfer_matrix, i, poles)
            else:
                idx = list(range(transfer_matrix.shape[1]))
                idx.pop(i)
                idx.pop(j - 1)
                transfer_matrix_not_i_j = transfer_matrix[:, idx]
                Q, _ = cupy.linalg.qr(transfer_matrix_not_i_j, mode='complete')
                if cupy.isreal(poles[i]):
                    assert cupy.isreal(poles[j]), 'mixing real and complex ' + 'in YT_real' + str(poles)
                    _YT_real(ker_pole, Q, transfer_matrix, i, j)
                else:
                    msg = 'mixing real and complex in YT_real' + str(poles)
                    assert ~cupy.isreal(poles[i]), msg
                    _YT_complex(ker_pole, Q, transfer_matrix, i, j)
        sq_spacing = sqrt(cupy.finfo(cupy.float64).eps)
        det_transfer_matrix = max((sq_spacing, cupy.abs(cupy.linalg.det(transfer_matrix))))
        cur_rtol = cupy.abs((det_transfer_matrix - det_transfer_matrixb) / det_transfer_matrix)
        if cur_rtol < rtol and det_transfer_matrix > sq_spacing:
            stop = True
        nb_try += 1
    return (stop, cur_rtol, nb_try)