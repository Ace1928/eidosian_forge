import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.ao.quantization.utils import (
def _combine_histograms(self, orig_hist: torch.Tensor, new_hist: torch.Tensor, upsample_rate: int, downsample_rate: int, start_idx: int, Nbins: int) -> torch.Tensor:
    upsampled_histogram = new_hist.repeat_interleave(upsample_rate)
    histogram_with_output_range = torch.zeros(Nbins * downsample_rate, device=orig_hist.device)
    histogram_with_output_range[start_idx:Nbins * upsample_rate + start_idx] = upsampled_histogram
    integral_histogram = torch.cumsum(histogram_with_output_range, 0, dtype=torch.double)[downsample_rate - 1::downsample_rate]
    shifted_integral_histogram = torch.zeros(Nbins, device=orig_hist.device)
    shifted_integral_histogram[1:Nbins] = integral_histogram[0:-1]
    interpolated_histogram = (integral_histogram - shifted_integral_histogram) / upsample_rate
    orig_hist = orig_hist + interpolated_histogram.to(torch.float)
    return orig_hist