import contextlib
import dataclasses
import math
import textwrap
from typing import Any, Dict, Optional
import torch
from torch import inf
def _tensor_str(self, indent):
    if self.numel() == 0:
        return '[]'
    if self.has_names():
        self = self.rename(None)
    summarize = self.numel() > PRINT_OPTS.threshold
    if self._is_zerotensor():
        self = self.clone()
    if self.is_neg():
        self = self.resolve_neg()
    if self.dtype in [torch.float16, torch.bfloat16, torch.float8_e5m2, torch.float8_e5m2fnuz, torch.float8_e4m3fn, torch.float8_e4m3fnuz]:
        self = self.float()
    if self.dtype is torch.complex32:
        self = self.cfloat()
    if self.dtype.is_complex:
        self = self.resolve_conj()
        real_formatter = _Formatter(get_summarized_data(self.real) if summarize else self.real)
        imag_formatter = _Formatter(get_summarized_data(self.imag) if summarize else self.imag)
        return _tensor_str_with_formatter(self, indent, summarize, real_formatter, imag_formatter)
    else:
        formatter = _Formatter(get_summarized_data(self) if summarize else self)
        return _tensor_str_with_formatter(self, indent, summarize, formatter)