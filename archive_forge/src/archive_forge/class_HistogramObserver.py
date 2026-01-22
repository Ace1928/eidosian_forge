import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.ao.quantization.utils import (
class HistogramObserver(UniformQuantizationObserverBase):
    """
    The module records the running histogram of tensor values along with
    min/max values. ``calculate_qparams`` will calculate scale and zero_point.

    Args:
        bins: Number of bins to use for the histogram
        upsample_rate: Factor by which the histograms are upsampled, this is
                       used to interpolate histograms with varying ranges across observations
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The scale and zero point are computed as follows:

    1. Create the histogram of the incoming inputs.
        The histogram is computed continuously, and the ranges per bin change
        with every new tensor observed.
    2. Search the distribution in the histogram for optimal min/max values.
        The search for the min/max values ensures the minimization of the
        quantization error with respect to the floating point model.
    3. Compute the scale and zero point the same way as in the
        :class:`~torch.ao.quantization.MinMaxObserver`
    """
    histogram: torch.Tensor
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self, bins: int=2048, upsample_rate: int=128, dtype: torch.dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False, quant_min=None, quant_max=None, factory_kwargs=None, eps=torch.finfo(torch.float32).eps, is_dynamic=False, **kwargs) -> None:
        if not is_per_tensor(qscheme):
            raise NotImplementedError("HistogramObserver's qscheme only support torch.per_tensor_symmetric                     and torch.per_tensor_affine.")
        if is_dynamic:
            raise NotImplementedError("HistogramObserver doesn't support dynamic quantization")
        super().__init__(dtype=dtype, qscheme=qscheme, reduce_range=reduce_range, quant_min=quant_min, quant_max=quant_max, factory_kwargs=factory_kwargs, eps=eps, is_dynamic=is_dynamic, **kwargs)
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.bins = bins
        self.register_buffer('histogram', torch.zeros(self.bins, **factory_kwargs))
        self.register_buffer('min_val', torch.tensor(float('inf'), **factory_kwargs))
        self.register_buffer('max_val', torch.tensor(float('-inf'), **factory_kwargs))
        self.dst_nbins = 2 ** torch.iinfo(self.dtype).bits
        self.upsample_rate = upsample_rate

    def _get_norm(self, delta_begin: torch.Tensor, delta_end: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        """
        Compute the norm of the values uniformaly distributed between
        delta_begin and delta_end.
        Currently only L2 norm is supported.

        norm = density * (integral_{begin, end} x^2)
             = density * (end^3 - begin^3) / 3
        """
        norm = (delta_end * delta_end * delta_end - delta_begin * delta_begin * delta_begin) / 3
        return density * norm

    def _compute_quantization_error(self, next_start_bin: int, next_end_bin: int):
        """
        Compute the quantization error if we use start_bin to end_bin as the
        min and max to do the quantization.
        """
        bin_width = (self.max_val.item() - self.min_val.item()) / self.bins
        dst_bin_width = bin_width * (next_end_bin - next_start_bin + 1) / self.dst_nbins
        if dst_bin_width == 0.0:
            return 0.0
        src_bin = torch.arange(self.bins, device=self.histogram.device)
        src_bin_begin = (src_bin - next_start_bin) * bin_width
        src_bin_end = src_bin_begin + bin_width
        dst_bin_of_begin = torch.clamp(torch.div(src_bin_begin, dst_bin_width, rounding_mode='floor'), 0, self.dst_nbins - 1)
        dst_bin_of_begin_center = (dst_bin_of_begin + 0.5) * dst_bin_width
        dst_bin_of_end = torch.clamp(torch.div(src_bin_end, dst_bin_width, rounding_mode='floor'), 0, self.dst_nbins - 1)
        density = self.histogram / bin_width
        norm = torch.zeros(self.bins, device=self.histogram.device)
        delta_begin = src_bin_begin - dst_bin_of_begin_center
        delta_end = dst_bin_width / 2
        norm += self._get_norm(delta_begin, torch.ones(self.bins, device=self.histogram.device) * delta_end, density)
        norm += (dst_bin_of_end - dst_bin_of_begin - 1) * self._get_norm(torch.tensor(-dst_bin_width / 2), torch.tensor(dst_bin_width / 2), density)
        dst_bin_of_end_center = dst_bin_of_end * dst_bin_width + dst_bin_width / 2
        delta_begin = -dst_bin_width / 2
        delta_end = src_bin_end - dst_bin_of_end_center
        norm += self._get_norm(torch.tensor(delta_begin), delta_end, density)
        return norm.sum().item()

    def _non_linear_param_search(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Non-linear parameter search.

        An approximation for L2 error minimization for selecting min/max.
        By selecting new min/max, we filter out outliers in input distribution.
        This follows the implementation of NormMinimization::NonlinearQuantizationParamsSearch in
        caffe2/quantization/server/norm_minimization.cc
        """
        assert self.histogram.size()[0] == self.bins, 'bins mismatch'
        bin_width = (self.max_val - self.min_val) / self.bins
        total = torch.sum(self.histogram).item()
        cSum = torch.cumsum(self.histogram, dim=0)
        stepsize = 1e-05
        alpha = 0.0
        beta = 1.0
        start_bin = 0
        end_bin = self.bins - 1
        norm_min = float('inf')
        while alpha < beta:
            next_alpha = alpha + stepsize
            next_beta = beta - stepsize
            l = start_bin
            r = end_bin
            while l < end_bin and cSum[l] < next_alpha * total:
                l = l + 1
            while r > start_bin and cSum[r] > next_beta * total:
                r = r - 1
            next_start_bin = start_bin
            next_end_bin = end_bin
            if l - start_bin > end_bin - r:
                next_start_bin = l
                alpha = next_alpha
            else:
                next_end_bin = r
                beta = next_beta
            if next_start_bin == start_bin and next_end_bin == end_bin:
                continue
            norm = self._compute_quantization_error(next_start_bin, next_end_bin)
            if norm > norm_min:
                break
            norm_min = norm
            start_bin = next_start_bin
            end_bin = next_end_bin
        new_min = self.min_val + bin_width * start_bin
        new_max = self.min_val + bin_width * (end_bin + 1)
        return (new_min, new_max)

    def _adjust_min_max(self, combined_min: torch.Tensor, combined_max: torch.Tensor, upsample_rate: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        downsample_rate = int(torch.ceil((combined_max - combined_min) * upsample_rate / (self.max_val - self.min_val)).item())
        e = downsample_rate * (self.max_val - self.min_val) / upsample_rate - (combined_max - combined_min)
        start_idx = int(torch.round((self.min_val - combined_min) * self.bins * upsample_rate / (self.max_val - self.min_val)).item())
        combined_max = combined_max + e
        combined_min = combined_min
        return (combined_min, combined_max, downsample_rate, start_idx)

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

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()
        min_val = self.min_val
        max_val = self.max_val
        same_values = min_val.item() == max_val.item()
        is_uninitialized = min_val == float('inf') and max_val == float('-inf')
        if is_uninitialized or same_values:
            min_val, max_val = torch.aminmax(x)
            self.min_val.resize_(min_val.shape)
            self.min_val.copy_(min_val)
            self.max_val.resize_(max_val.shape)
            self.max_val.copy_(max_val)
            assert min_val.numel() == 1 and max_val.numel() == 1, 'histogram min/max values must be scalar.'
            torch.histc(x, self.bins, min=min_val, max=max_val, out=self.histogram)
        else:
            new_min, new_max = torch.aminmax(x)
            combined_min = torch.min(new_min, min_val)
            combined_max = torch.max(new_max, max_val)
            combined_min, combined_max, downsample_rate, start_idx = self._adjust_min_max(combined_min, combined_max, self.upsample_rate)
            assert combined_min.numel() == 1 and combined_max.numel() == 1, 'histogram min/max values must be scalar.'
            combined_min, combined_max = (combined_min.detach(), combined_max.detach())
            combined_histogram = torch.histc(x, self.bins, min=combined_min, max=combined_max)
            if combined_min == min_val and combined_max == max_val:
                combined_histogram += self.histogram
            else:
                combined_histogram = self._combine_histograms(combined_histogram, self.histogram, self.upsample_rate, downsample_rate, start_idx, self.bins)
            self.histogram.detach_().resize_(combined_histogram.shape)
            self.histogram.copy_(combined_histogram)
            self.min_val.detach_().resize_(combined_min.shape)
            self.min_val.copy_(combined_min)
            self.max_val.detach_().resize_(combined_max.shape)
            self.max_val.copy_(combined_max)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        is_uninitialized = self.min_val == float('inf') and self.max_val == float('-inf')
        if is_uninitialized:
            warnings.warn('must run observer before calling calculate_qparams.                                    Returning default scale and zero point ')
            return (torch.tensor([1.0], device=self.min_val.device.type), torch.tensor([0], device=self.min_val.device.type))
        assert self.bins == len(self.histogram), 'The number of bins in histogram should be equal to the number of bins supplied while making this observer'
        new_min, new_max = self._non_linear_param_search()
        return self._calculate_qparams(new_min, new_max)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'min_val'] = self.min_val
        destination[prefix + 'max_val'] = self.max_val

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version < 3:
            min_val_name, max_val_name = (prefix + 'min_val', prefix + 'max_val')
            if min_val_name in state_dict:
                if state_dict[min_val_name].shape == torch.Size([0]):
                    state_dict[min_val_name] = torch.tensor(float('inf'))
            if max_val_name in state_dict:
                if state_dict[max_val_name].shape == torch.Size([0]):
                    state_dict[max_val_name] = torch.tensor(float('-inf'))
        local_state = ['min_val', 'max_val']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def extra_repr(self):
        return f'min_val={self.min_val}, max_val={self.max_val}'