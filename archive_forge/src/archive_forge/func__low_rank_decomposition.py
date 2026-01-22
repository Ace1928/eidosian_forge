import logging
from typing import Union
import torch
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
def _low_rank_decomposition(weight, reduced_rank=32):
    """
    :param weight: The matrix to decompose, of shape (H, W) :param reduced_rank: the final rank :return:
    """
    matrix_dimension = len(weight.size())
    if matrix_dimension != 2:
        raise ValueError(f'Only support 2D matrix, but your input has {matrix_dimension} dimensions.')
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    L = U @ torch.sqrt(torch.diag(S)[:, 0:reduced_rank])
    R = torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh
    return {'L': L, 'R': R, 'U': U, 'S': S, 'Vh': Vh, 'reduced_rank': reduced_rank}