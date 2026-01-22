import warnings
from typing import Iterable, List, NamedTuple, Tuple, Union
import torch
from torch import Tensor
from ... import _VF
from ..._jit_internal import Optional
class PackedSequence(PackedSequence_):
    """Holds the data and list of :attr:`batch_sizes` of a packed sequence.

    All RNN modules accept packed sequences as inputs.

    Note:
        Instances of this class should never be created manually. They are meant
        to be instantiated by functions like :func:`pack_padded_sequence`.

        Batch sizes represent the number elements at each sequence step in
        the batch, not the varying sequence lengths passed to
        :func:`pack_padded_sequence`.  For instance, given data ``abc`` and ``x``
        the :class:`PackedSequence` would contain data ``axbc`` with
        ``batch_sizes=[2,1,1]``.

    Attributes:
        data (Tensor): Tensor containing packed sequence
        batch_sizes (Tensor): Tensor of integers holding
            information about the batch size at each sequence step
        sorted_indices (Tensor, optional): Tensor of integers holding how this
            :class:`PackedSequence` is constructed from sequences.
        unsorted_indices (Tensor, optional): Tensor of integers holding how this
            to recover the original sequences with correct order.

    .. note::
        :attr:`data` can be on arbitrary device and of arbitrary dtype.
        :attr:`sorted_indices` and :attr:`unsorted_indices` must be ``torch.int64``
        tensors on the same device as :attr:`data`.

        However, :attr:`batch_sizes` should always be a CPU ``torch.int64`` tensor.

        This invariant is maintained throughout :class:`PackedSequence` class,
        and all functions that construct a `:class:PackedSequence` in PyTorch
        (i.e., they only pass in tensors conforming to this constraint).

    """

    def __new__(cls, data, batch_sizes=None, sorted_indices=None, unsorted_indices=None):
        return super().__new__(cls, *_packed_sequence_init_args(data, batch_sizes, sorted_indices, unsorted_indices))

    def pin_memory(self):
        return type(self)(self.data.pin_memory(), self.batch_sizes, bind(self.sorted_indices, lambda t: t.pin_memory()), bind(self.unsorted_indices, lambda t: t.pin_memory()))

    def cuda(self, *args, **kwargs):
        ex = torch.tensor((), dtype=self.data.dtype, device=self.data.device).to(*args, **kwargs)
        if ex.is_cuda:
            return self.to(*args, **kwargs)
        return self.to(*args, device='cuda', **kwargs)

    def cpu(self, *args, **kwargs):
        ex = torch.tensor((), dtype=self.data.dtype, device=self.data.device).to(*args, **kwargs)
        if ex.device.type == 'cpu':
            return self.to(*args, **kwargs)
        return self.to(*args, device='cpu', **kwargs)

    def double(self):
        return self.to(dtype=torch.double)

    def float(self):
        return self.to(dtype=torch.float)

    def half(self):
        return self.to(dtype=torch.half)

    def long(self):
        return self.to(dtype=torch.long)

    def int(self):
        return self.to(dtype=torch.int)

    def short(self):
        return self.to(dtype=torch.short)

    def char(self):
        return self.to(dtype=torch.int8)

    def byte(self):
        return self.to(dtype=torch.uint8)

    def to(self, *args, **kwargs):
        """Perform dtype and/or device conversion on `self.data`.

        It has similar signature as :meth:`torch.Tensor.to`, except optional
        arguments like `non_blocking` and `copy` should be passed as kwargs,
        not args, or they will not apply to the index tensors.

        .. note::

            If the ``self.data`` Tensor already has the correct :class:`torch.dtype`
            and :class:`torch.device`, then ``self`` is returned.
            Otherwise, returns a copy with the desired configuration.
        """
        data = self.data.to(*args, **kwargs)
        if data is self.data:
            return self
        else:
            kwargs = dict(filter(lambda t: t[0] != 'device' and t[0] != 'dtype', kwargs.items()))
            sorted_indices = bind(self.sorted_indices, lambda t: t.to(data.device, **kwargs))
            unsorted_indices = bind(self.unsorted_indices, lambda t: t.to(data.device, **kwargs))
            return type(self)(data, self.batch_sizes, sorted_indices, unsorted_indices)

    @property
    def is_cuda(self):
        """Return true if `self.data` stored on a gpu."""
        return self.data.is_cuda

    def is_pinned(self):
        """Return true if `self.data` stored on in pinned memory."""
        return self.data.is_pinned()