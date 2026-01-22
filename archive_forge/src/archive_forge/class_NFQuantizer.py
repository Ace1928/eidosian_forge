import logging
from typing import Union
import torch
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
class NFQuantizer:

    def __init__(self, num_bits=2, device='cuda', method='normal', block_size=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_bits = num_bits
        self.device = device
        self.method = method
        self.block_size = block_size
        if self.method == 'normal':
            self.norm_lookup_table = self.create_normal_map(num_bits=self.num_bits)
            self.norm_lookup_table = self.norm_lookup_table.to(device)
        elif self.method == 'uniform':
            self.norm_lookup_table = self.create_uniform_map(num_bits=self.num_bits)
            self.norm_lookup_table = self.norm_lookup_table.to(device)
        else:
            raise NotImplementedError('Other quantization methods not supported yet.')

    @staticmethod
    def create_uniform_map(symmetric=False, num_bits=4):
        if symmetric:
            negative = torch.linspace(-1, 0, 2 ** (num_bits - 1))
            positive = torch.linspace(0, 1, 2 ** (num_bits - 1))
            table = torch.cat([negative, positive[1:]])
        else:
            table = torch.linspace(-1, 1, 2 ** num_bits)
        return table

    @staticmethod
    def create_normal_map(offset=0.9677083, symmetric=False, num_bits=2):
        try:
            from scipy.stats import norm
        except ImportError:
            raise ImportError("The required package 'scipy' is not installed. Please install it to continue.")
        variations = 2 ** num_bits
        if symmetric:
            v = norm.ppf(torch.linspace(1 - offset, offset, variations + 1)).tolist()
            values = []
            for index in range(len(v) - 1):
                values.append(0.5 * v[index] + 0.5 * v[index + 1])
            v = values
        else:
            v1 = norm.ppf(torch.linspace(offset, 0.5, variations // 2 + 1)[:-1]).tolist()
            v2 = [0]
            v3 = (-norm.ppf(torch.linspace(offset, 0.5, variations // 2)[:-1])).tolist()
            v = v1 + v2 + v3
        values = torch.Tensor(v)
        values = values.sort().values
        values /= values.max()
        return values

    def quantize_tensor(self, weight):
        max_abs = torch.abs(weight).max()
        weight_normed = weight / max_abs
        weight_normed_expanded = weight_normed.unsqueeze(-1)
        L_reshaped = torch.tensor(self.norm_lookup_table).reshape(1, -1)
        abs_diff = torch.abs(weight_normed_expanded - L_reshaped)
        qweight = torch.argmin(abs_diff, dim=-1)
        return (qweight, max_abs)

    def dequantize_tensor(self, qweight, max_abs):
        qweight_flatten = qweight.flatten()
        weight_normed = self.norm_lookup_table[qweight_flatten]
        weight = weight_normed * max_abs
        weight = weight.reshape(qweight.shape)
        return weight

    def quantize_block(self, weight):
        if len(weight.shape) != 2:
            raise ValueError(f'Only support 2D matrix, but your input has {len(weight.shape)} dimensions.')
        if weight.shape[0] * weight.shape[1] % self.block_size != 0:
            raise ValueError(f'Weight with shape ({weight.shape[0]} x {weight.shape[1]}) is not dividable by block size {self.block_size}.')
        M, N = weight.shape
        device = weight.device
        weight_flatten = weight.flatten()
        weight_block = weight_flatten.reshape(-1, self.block_size)
        if self.method == 'normal':
            weight_max = weight_block.abs().max(dim=-1)[0]
        elif self.method == 'uniform':
            weight_max = weight_block.mean(dim=-1) + 2.5 * weight_block.std(dim=-1)
        else:
            raise NotImplementedError('Method not supported yet.')
        weight_max = weight_max.unsqueeze(-1)
        weight_divabs = weight_block / weight_max
        weight_divabs = weight_divabs.unsqueeze(-1)
        L_reshaped = self.norm_lookup_table.reshape(1, -1)
        abs_diff = torch.abs(weight_divabs - L_reshaped)
        qweight = torch.argmin(abs_diff, dim=-1)
        qweight = qweight.reshape(-1, 8 // self.num_bits)
        qweight_pack = torch.zeros((M * N // 8 * self.num_bits, 1), dtype=torch.uint8, device=device)
        for i in range(8 // self.num_bits):
            qweight[:, i] = qweight[:, i] << i * self.num_bits
            qweight_pack[:, 0] |= qweight[:, i]
        return (qweight_pack, weight_max, weight.shape)

    def dequantize_block(self, qweight, weight_max, weight_shape):
        device = qweight.device
        weight = torch.zeros((qweight.shape[0], 8 // self.num_bits), dtype=torch.float32, device=device)
        for i in range(8 // self.num_bits):
            lookup_table_idx = qweight.to(torch.long) % 2 ** self.num_bits
            lookup_table_idx = lookup_table_idx.to(torch.long)
            weight[:, i] = self.norm_lookup_table[lookup_table_idx].squeeze()
            qweight = qweight >> self.num_bits
        weight_block = weight.reshape(-1, self.block_size)
        weight = weight_block * weight_max
        weight = weight.reshape(weight_shape)
        return weight