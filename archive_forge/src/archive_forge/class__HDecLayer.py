import math
import typing as tp
from typing import Any, Dict, List, Optional
import torch
from torch import nn
from torch.nn import functional as F
class _HDecLayer(torch.nn.Module):
    """Decoder layer. This used both by the time and the frequency branches.
    Args:
        chin (int): number of input channels.
        chout (int): number of output channels.
        last (bool, optional): whether current layer is final layer (Default: ``False``)
        kernel_size (int, optional): Kernel size for encoder (Default: 8)
        stride (int): Stride for encoder layer (Default: 4)
        norm_groups (int, optional): number of groups for group norm. (Default: 1)
        empty (bool, optional): used to make a layer with just the first conv. this is used
            before merging the time and freq. branches. (Default: ``False``)
        freq (bool, optional): boolean for whether conv layer is for frequency (Default: ``True``)
        norm_type (str, optional): Norm type, either ``group_norm `` or ``none`` (Default: ``group_norm``)
        context (int, optional): context size for the 1x1 conv. (Default: 1)
        dconv_kw (Dict[str, Any] or None, optional): dictionary of kwargs for the DConv class. (Default: ``None``)
        pad (bool, optional): true to pad the input. Padding is done so that the output size is
            always the input size / stride. (Default: ``True``)
    """

    def __init__(self, chin: int, chout: int, last: bool=False, kernel_size: int=8, stride: int=4, norm_groups: int=1, empty: bool=False, freq: bool=True, norm_type: str='group_norm', context: int=1, dconv_kw: Optional[Dict[str, Any]]=None, pad: bool=True):
        super().__init__()
        if dconv_kw is None:
            dconv_kw = {}
        norm_fn = lambda d: nn.Identity()
        if norm_type == 'group_norm':
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)
        if pad:
            if (kernel_size - stride) % 2 != 0:
                raise ValueError('Kernel size and stride do not align')
            pad = (kernel_size - stride) // 2
        else:
            pad = 0
        self.pad = pad
        self.last = last
        self.freq = freq
        self.chin = chin
        self.empty = empty
        self.stride = stride
        self.kernel_size = kernel_size
        klass = nn.Conv1d
        klass_tr = nn.ConvTranspose1d
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            klass = nn.Conv2d
            klass_tr = nn.ConvTranspose2d
        self.conv_tr = klass_tr(chin, chout, kernel_size, stride)
        self.norm2 = norm_fn(chout)
        if self.empty:
            self.rewrite = nn.Identity()
            self.norm1 = nn.Identity()
        else:
            self.rewrite = klass(chin, 2 * chin, 1 + 2 * context, 1, context)
            self.norm1 = norm_fn(2 * chin)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor], length):
        """Forward pass for decoding layer.

        Size depends on whether frequency or time

        Args:
            x (torch.Tensor): tensor input of shape `(B, C, F, T)` for frequency and shape
                `(B, C, T)` for time
            skip (torch.Tensor, optional): on first layer, separate frequency and time branches using param
                (default: ``None``)
            length (int): Size of tensor for output

        Returns:
            (Tensor, Tensor):
                Tensor
                    output tensor after decoder layer of shape `(B, C, F * stride, T)` for frequency domain except last
                        frequency layer shape is `(B, C, kernel_size, T)`. Shape is `(B, C, stride * T)`
                        for time domain.
                Tensor
                    contains the output just before final transposed convolution, which is used when the
                        freq. and time branch separate. Otherwise, does not matter. Shape is
                        `(B, C, F, T)` for frequency and `(B, C, T)` for time.
        """
        if self.freq and x.dim() == 3:
            B, C, T = x.shape
            x = x.view(B, self.chin, -1, T)
        if not self.empty:
            x = x + skip
            y = F.glu(self.norm1(self.rewrite(x)), dim=1)
        else:
            y = x
            if skip is not None:
                raise ValueError('Skip must be none when empty is true.')
        z = self.norm2(self.conv_tr(y))
        if self.freq:
            if self.pad:
                z = z[..., self.pad:-self.pad, :]
        else:
            z = z[..., self.pad:self.pad + length]
            if z.shape[-1] != length:
                raise ValueError('Last index of z must be equal to length')
        if not self.last:
            z = F.gelu(z)
        return (z, y)