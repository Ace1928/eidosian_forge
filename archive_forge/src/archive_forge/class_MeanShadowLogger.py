import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
import torch.ao.quantization
import torch.ao.ns._numeric_suite as ns
class MeanShadowLogger(ns.Logger):
    """Mean Logger for a Shadow module.

    A logger for a Shadow module whose purpose is to record the rolling mean
    of the data passed to the floating point and quantized models
    """

    def __init__(self):
        """Set up initial values for float and quantized stats, count, float sum, and quant sum."""
        super().__init__()
        self.stats['float'] = None
        self.stats['quantized'] = None
        self.count = 0
        self.float_sum = None
        self.quant_sum = None

    def forward(self, x, y):
        """Compute the average of quantized and floating-point data from modules.

        The inputs x,y are output data from the quantized and floating-point modules.
        x is for the quantized module, y is for the floating point module
        """
        if x.is_quantized:
            x = x.dequantize()
        self.count += 1
        if self.stats['quantized'] is None:
            self.stats['quantized'] = x
            self.quant_sum = x
        else:
            self.quant_sum += x
            self.stats['quantized'] = self.quant_sum / self.count
        if self.stats['float'] is None:
            self.stats['float'] = y
            self.float_sum = y
        else:
            self.float_sum += y
            self.stats['float'] = self.float_sum / self.count

    def clear(self):
        self.stats['float'] = None
        self.stats['quantized'] = None
        self.count = 0
        self.float_sum = None
        self.quant_sum = None