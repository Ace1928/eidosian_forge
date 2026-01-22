import numbers
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Tuple
import torch
class PruningContainer(BasePruningMethod):
    """Container holding a sequence of pruning methods for iterative pruning.

    Keeps track of the order in which pruning methods are applied and handles
    combining successive pruning calls.

    Accepts as argument an instance of a BasePruningMethod or an iterable of
    them.
    """

    def __init__(self, *args):
        self._pruning_methods: Tuple[BasePruningMethod, ...] = tuple()
        if not isinstance(args, Iterable):
            self._tensor_name = args._tensor_name
            self.add_pruning_method(args)
        elif len(args) == 1:
            self._tensor_name = args[0]._tensor_name
            self.add_pruning_method(args[0])
        else:
            for method in args:
                self.add_pruning_method(method)

    def add_pruning_method(self, method):
        """Add a child pruning ``method`` to the container.

        Args:
            method (subclass of BasePruningMethod): child pruning method
                to be added to the container.
        """
        if not isinstance(method, BasePruningMethod) and method is not None:
            raise TypeError(f'{type(method)} is not a BasePruningMethod subclass')
        elif method is not None and self._tensor_name != method._tensor_name:
            raise ValueError(f"Can only add pruning methods acting on the parameter named '{self._tensor_name}' to PruningContainer {self}." + f" Found '{method._tensor_name}'")
        self._pruning_methods += (method,)

    def __len__(self):
        return len(self._pruning_methods)

    def __iter__(self):
        return iter(self._pruning_methods)

    def __getitem__(self, idx):
        return self._pruning_methods[idx]

    def compute_mask(self, t, default_mask):
        """Apply the latest ``method`` by computing the new partial masks and returning its combination with the ``default_mask``.

        The new partial mask should be computed on the entries or channels
        that were not zeroed out by the ``default_mask``.
        Which portions of the tensor ``t`` the new mask will be calculated from
        depends on the ``PRUNING_TYPE`` (handled by the type handler):

        * for 'unstructured', the mask will be computed from the raveled
          list of nonmasked entries;

        * for 'structured', the mask will be computed from the nonmasked
          channels in the tensor;

        * for 'global', the mask will be computed across all entries.

        Args:
            t (torch.Tensor): tensor representing the parameter to prune
                (of same dimensions as ``default_mask``).
            default_mask (torch.Tensor): mask from previous pruning iteration.

        Returns:
            mask (torch.Tensor): new mask that combines the effects
            of the ``default_mask`` and the new mask from the current
            pruning ``method`` (of same dimensions as ``default_mask`` and
            ``t``).
        """

        def _combine_masks(method, t, mask):
            """Combine the masks from all pruning methods and returns a new mask.

            Args:
                method (a BasePruningMethod subclass): pruning method
                    currently being applied.
                t (torch.Tensor): tensor representing the parameter to prune
                    (of same dimensions as mask).
                mask (torch.Tensor): mask from previous pruning iteration

            Returns:
                new_mask (torch.Tensor): new mask that combines the effects
                    of the old mask and the new mask from the current
                    pruning method (of same dimensions as mask and t).
            """
            new_mask = mask
            new_mask = new_mask.to(dtype=t.dtype)
            if method.PRUNING_TYPE == 'unstructured':
                slc = mask == 1
            elif method.PRUNING_TYPE == 'structured':
                if not hasattr(method, 'dim'):
                    raise AttributeError('Pruning methods of PRUNING_TYPE "structured" need to have the attribute `dim` defined.')
                n_dims = t.dim()
                dim = method.dim
                if dim < 0:
                    dim = n_dims + dim
                if dim < 0:
                    raise IndexError(f'Index is out of bounds for tensor with dimensions {n_dims}')
                keep_channel = mask.sum(dim=[d for d in range(n_dims) if d != dim]) != 0
                slc = [slice(None)] * n_dims
                slc[dim] = keep_channel
            elif method.PRUNING_TYPE == 'global':
                n_dims = len(t.shape)
                slc = [slice(None)] * n_dims
            else:
                raise ValueError(f'Unrecognized PRUNING_TYPE {method.PRUNING_TYPE}')
            partial_mask = method.compute_mask(t[slc], default_mask=mask[slc])
            new_mask[slc] = partial_mask.to(dtype=new_mask.dtype)
            return new_mask
        method = self._pruning_methods[-1]
        mask = _combine_masks(method, t, default_mask)
        return mask