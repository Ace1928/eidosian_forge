import math
import numbers
import os
import cupy
from ._util import _get_inttype
def _distance_tranform_arg_check(distances_out, indices_out, return_distances, return_indices):
    """Raise a RuntimeError if the arguments are invalid"""
    error_msgs = []
    if not return_distances and (not return_indices):
        error_msgs.append('at least one of return_distances/return_indices must be True')
    if distances_out and (not return_distances):
        error_msgs.append('return_distances must be True if distances is supplied')
    if indices_out and (not return_indices):
        error_msgs.append('return_indices must be True if indices is supplied')
    if error_msgs:
        raise RuntimeError(', '.join(error_msgs))