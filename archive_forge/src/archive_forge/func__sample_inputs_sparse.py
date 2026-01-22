import os
import torch
from torch.testing import make_tensor  # noqa: F401
from torch.testing._internal.opinfo.core import (  # noqa: F401
def _sample_inputs_sparse(sample_inputs, maybe_failing_sample_inputs, validate_sample_input, op_info, *args, **kwargs):
    check_validate = os.environ.get('PYTORCH_TEST_CHECK_VALIDATE_SPARSE_SAMPLES', '0') == '1'
    for sample in sample_inputs(op_info, *args, **kwargs):
        sample = validate_sample_input(op_info, sample, check_validate=check_validate)
        if isinstance(sample, SampleInput):
            yield sample
    for sample in maybe_failing_sample_inputs(op_info, *args, **kwargs):
        sample = validate_sample_input(op_info, sample, check_validate=check_validate)
        if isinstance(sample, SampleInput):
            yield sample