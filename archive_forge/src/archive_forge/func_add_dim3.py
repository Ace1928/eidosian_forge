import argparse
import fnmatch
import re
import shutil
import sys
import os
from . import constants
from .cuda_to_hip_mappings import CUDA_TO_HIP_MAPPINGS
from .cuda_to_hip_mappings import MATH_TRANSPILATIONS
from typing import Dict, List, Iterator, Optional
from collections.abc import Mapping, Iterable
from enum import Enum
def add_dim3(kernel_string, cuda_kernel):
    """adds dim3() to the second and third arguments in the kernel launch"""
    count = 0
    closure = 0
    kernel_string = kernel_string.replace('<<<', '').replace('>>>', '')
    arg_locs: List[Dict[str, int]] = [{} for _ in range(2)]
    arg_locs[count]['start'] = 0
    for ind, c in enumerate(kernel_string):
        if count > 1:
            break
        if c == '(':
            closure += 1
        elif c == ')':
            closure -= 1
        if (c == ',' or ind == len(kernel_string) - 1) and closure == 0:
            arg_locs[count]['end'] = ind + (c != ',')
            count += 1
            if count < 2:
                arg_locs[count]['start'] = ind + 1
    first_arg_raw = kernel_string[arg_locs[0]['start']:arg_locs[0]['end'] + 1]
    second_arg_raw = kernel_string[arg_locs[1]['start']:arg_locs[1]['end']]
    first_arg_clean = kernel_string[arg_locs[0]['start']:arg_locs[0]['end']].replace('\n', '').strip(' ')
    second_arg_clean = kernel_string[arg_locs[1]['start']:arg_locs[1]['end']].replace('\n', '').strip(' ')
    first_arg_dim3 = f'dim3({first_arg_clean})'
    second_arg_dim3 = f'dim3({second_arg_clean})'
    first_arg_raw_dim3 = first_arg_raw.replace(first_arg_clean, first_arg_dim3)
    second_arg_raw_dim3 = second_arg_raw.replace(second_arg_clean, second_arg_dim3)
    cuda_kernel = cuda_kernel.replace(first_arg_raw + second_arg_raw, first_arg_raw_dim3 + second_arg_raw_dim3)
    return cuda_kernel