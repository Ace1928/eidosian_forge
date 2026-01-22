import contextlib
import dataclasses
import math
import textwrap
from typing import Any, Dict, Optional
import torch
from torch import inf
class _Formatter:

    def __init__(self, tensor):
        self.floating_dtype = tensor.dtype.is_floating_point
        self.int_mode = True
        self.sci_mode = False
        self.max_width = 1
        with torch.no_grad():
            tensor_view = tensor.reshape(-1)
        if not self.floating_dtype:
            for value in tensor_view:
                value_str = f'{value}'
                self.max_width = max(self.max_width, len(value_str))
        else:
            nonzero_finite_vals = torch.masked_select(tensor_view, torch.isfinite(tensor_view) & tensor_view.ne(0))
            if nonzero_finite_vals.numel() == 0:
                return
            nonzero_finite_abs = tensor_totype(nonzero_finite_vals.abs())
            nonzero_finite_min = tensor_totype(nonzero_finite_abs.min())
            nonzero_finite_max = tensor_totype(nonzero_finite_abs.max())
            for value in nonzero_finite_vals:
                if value != torch.ceil(value):
                    self.int_mode = False
                    break
            if self.int_mode:
                if nonzero_finite_max / nonzero_finite_min > 1000.0 or nonzero_finite_max > 100000000.0:
                    self.sci_mode = True
                    for value in nonzero_finite_vals:
                        value_str = f'{{:.{PRINT_OPTS.precision}e}}'.format(value)
                        self.max_width = max(self.max_width, len(value_str))
                else:
                    for value in nonzero_finite_vals:
                        value_str = f'{value:.0f}'
                        self.max_width = max(self.max_width, len(value_str) + 1)
            elif nonzero_finite_max / nonzero_finite_min > 1000.0 or nonzero_finite_max > 100000000.0 or nonzero_finite_min < 0.0001:
                self.sci_mode = True
                for value in nonzero_finite_vals:
                    value_str = f'{{:.{PRINT_OPTS.precision}e}}'.format(value)
                    self.max_width = max(self.max_width, len(value_str))
            else:
                for value in nonzero_finite_vals:
                    value_str = f'{{:.{PRINT_OPTS.precision}f}}'.format(value)
                    self.max_width = max(self.max_width, len(value_str))
        if PRINT_OPTS.sci_mode is not None:
            self.sci_mode = PRINT_OPTS.sci_mode

    def width(self):
        return self.max_width

    def format(self, value):
        if self.floating_dtype:
            if self.sci_mode:
                ret = f'{{:{self.max_width}.{PRINT_OPTS.precision}e}}'.format(value)
            elif self.int_mode:
                ret = f'{value:.0f}'
                if not (math.isinf(value) or math.isnan(value)):
                    ret += '.'
            else:
                ret = f'{{:.{PRINT_OPTS.precision}f}}'.format(value)
        else:
            ret = f'{value}'
        return (self.max_width - len(ret)) * ' ' + ret