import inspect
import os
from argparse import ArgumentParser, Namespace
from importlib import import_module
import huggingface_hub
import numpy as np
from packaging import version
from .. import (
from ..utils import TF2_WEIGHTS_INDEX_NAME, TF2_WEIGHTS_NAME, logging
from . import BaseTransformersCLICommand
def _find_pt_tf_differences(pt_out, tf_out, differences, attr_name=''):
    if isinstance(pt_out, torch.Tensor):
        tensor_difference = np.max(np.abs(pt_out.numpy() - tf_out.numpy()))
        differences[attr_name] = tensor_difference
    else:
        root_name = attr_name
        for i, pt_item in enumerate(pt_out):
            if isinstance(pt_item, str):
                branch_name = root_name + pt_item
                tf_item = tf_out[pt_item]
                pt_item = pt_out[pt_item]
            else:
                branch_name = root_name + f'[{i}]'
                tf_item = tf_out[i]
            differences = _find_pt_tf_differences(pt_item, tf_item, differences, branch_name)
    return differences