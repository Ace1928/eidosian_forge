import contextlib
from collections.abc import Iterable
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple, Type, Union
from torch.utils.data.dataloader import _BaseDataLoaderIter, _MultiProcessingDataLoaderIter
from typing_extensions import Self, TypedDict, override
from lightning_fabric.utilities.data import sized_len
from lightning_fabric.utilities.types import _Stateful
from pytorch_lightning.utilities._pytree import _map_and_unflatten, _tree_flatten, tree_unflatten
class CombinedLoader(Iterable):
    """Combines different iterables under specific sampling modes.

    Args:
        iterables: the iterable or collection of iterables to sample from.
        mode: the mode to use. The following modes are supported:

            * ``min_size``: stops after the shortest iterable (the one with the lowest number of items) is done.
            * ``max_size_cycle``: stops after the longest iterable (the one with most items) is done, while cycling
              through the rest of the iterables.
            * ``max_size``: stops after the longest iterable (the one with most items) is done, while returning None
              for the exhausted iterables.
            * ``sequential``: completely consumes each iterable sequentially, and returns a triplet
              ``(data, idx, iterable_idx)``

    Examples:
        >>> from torch.utils.data import DataLoader
        >>> iterables = {'a': DataLoader(range(6), batch_size=4),
        ...              'b': DataLoader(range(15), batch_size=5)}
        >>> combined_loader = CombinedLoader(iterables, 'max_size_cycle')
        >>> _ = iter(combined_loader)
        >>> len(combined_loader)
        3
        >>> for batch, batch_idx, dataloader_idx in combined_loader:
        ...     print(f"{batch}, {batch_idx=}, {dataloader_idx=}")
        {'a': tensor([0, 1, 2, 3]), 'b': tensor([0, 1, 2, 3, 4])}, batch_idx=0, dataloader_idx=0
        {'a': tensor([4, 5]), 'b': tensor([5, 6, 7, 8, 9])}, batch_idx=1, dataloader_idx=0
        {'a': tensor([0, 1, 2, 3]), 'b': tensor([10, 11, 12, 13, 14])}, batch_idx=2, dataloader_idx=0

        >>> combined_loader = CombinedLoader(iterables, 'max_size')
        >>> _ = iter(combined_loader)
        >>> len(combined_loader)
        3
        >>> for batch, batch_idx, dataloader_idx in combined_loader:
        ...     print(f"{batch}, {batch_idx=}, {dataloader_idx=}")
        {'a': tensor([0, 1, 2, 3]), 'b': tensor([0, 1, 2, 3, 4])}, batch_idx=0, dataloader_idx=0
        {'a': tensor([4, 5]), 'b': tensor([5, 6, 7, 8, 9])}, batch_idx=1, dataloader_idx=0
        {'a': None, 'b': tensor([10, 11, 12, 13, 14])}, batch_idx=2, dataloader_idx=0

        >>> combined_loader = CombinedLoader(iterables, 'min_size')
        >>> _ = iter(combined_loader)
        >>> len(combined_loader)
        2
        >>> for batch, batch_idx, dataloader_idx in combined_loader:
        ...     print(f"{batch}, {batch_idx=}, {dataloader_idx=}")
        {'a': tensor([0, 1, 2, 3]), 'b': tensor([0, 1, 2, 3, 4])}, batch_idx=0, dataloader_idx=0
        {'a': tensor([4, 5]), 'b': tensor([5, 6, 7, 8, 9])}, batch_idx=1, dataloader_idx=0

        >>> combined_loader = CombinedLoader(iterables, 'sequential')
        >>> _ = iter(combined_loader)
        >>> len(combined_loader)
        5
        >>> for batch, batch_idx, dataloader_idx in combined_loader:
        ...     print(f"{batch}, {batch_idx=}, {dataloader_idx=}")
        tensor([0, 1, 2, 3]), batch_idx=0, dataloader_idx=0
        tensor([4, 5]), batch_idx=1, dataloader_idx=0
        tensor([0, 1, 2, 3, 4]), batch_idx=0, dataloader_idx=1
        tensor([5, 6, 7, 8, 9]), batch_idx=1, dataloader_idx=1
        tensor([10, 11, 12, 13, 14]), batch_idx=2, dataloader_idx=1

    """

    def __init__(self, iterables: Any, mode: _LITERAL_SUPPORTED_MODES='min_size') -> None:
        if mode not in _SUPPORTED_MODES:
            raise ValueError(f'Unsupported mode {mode!r}, please select one of: {list(_SUPPORTED_MODES)}.')
        self._iterables = iterables
        self._flattened, self._spec = _tree_flatten(iterables)
        self._mode = mode
        self._iterator: Optional[_ModeIterator] = None
        self._limits: Optional[List[Union[int, float]]] = None

    @property
    def iterables(self) -> Any:
        """Return the original collection of iterables."""
        return self._iterables

    @property
    def sampler(self) -> Any:
        """Return a collections of samplers extracted from iterables."""
        return _map_and_unflatten(lambda x: getattr(x, 'sampler', None), self.flattened, self._spec)

    @property
    def batch_sampler(self) -> Any:
        """Return a collections of batch samplers extracted from iterables."""
        return _map_and_unflatten(lambda x: getattr(x, 'batch_sampler', None), self.flattened, self._spec)

    @property
    def flattened(self) -> List[Any]:
        """Return the flat list of iterables."""
        return self._flattened

    @flattened.setter
    def flattened(self, flattened: List[Any]) -> None:
        """Setter to conveniently update the list of iterables."""
        if len(flattened) != len(self._flattened):
            raise ValueError(f'Mismatch in flattened length ({len(flattened)}) and existing length ({len(self._flattened)})')
        self._iterables = tree_unflatten(flattened, self._spec)
        self._flattened = flattened

    @property
    def limits(self) -> Optional[List[Union[int, float]]]:
        """Optional limits per iterator."""
        return self._limits

    @limits.setter
    def limits(self, limits: Optional[Union[int, float, List[Union[int, float]]]]) -> None:
        if isinstance(limits, (int, float)):
            limits = [limits] * len(self.flattened)
        elif isinstance(limits, list) and len(limits) != len(self.flattened):
            raise ValueError(f'Mismatch in number of limits ({len(limits)}) and number of iterables ({len(self.flattened)})')
        self._limits = limits

    def __next__(self) -> _ITERATOR_RETURN:
        assert self._iterator is not None
        out = next(self._iterator)
        if isinstance(self._iterator, _Sequential):
            return out
        out, batch_idx, dataloader_idx = out
        return (tree_unflatten(out, self._spec), batch_idx, dataloader_idx)

    @override
    def __iter__(self) -> Self:
        cls = _SUPPORTED_MODES[self._mode]['iterator']
        iterator = cls(self.flattened, self._limits)
        iter(iterator)
        self._iterator = iterator
        return self

    def __len__(self) -> int:
        """Compute the number of batches."""
        if self._iterator is None:
            raise RuntimeError('Please call `iter(combined_loader)` first.')
        return len(self._iterator)

    def reset(self) -> None:
        """Reset the state and shutdown any workers."""
        if self._iterator is not None:
            self._iterator.reset()
            self._iterator = None
        for iterable in self.flattened:
            _shutdown_workers_and_reset_iterator(iterable)

    def _dataset_length(self) -> int:
        """Compute the total length of the datasets according to the current mode."""
        datasets = [getattr(dl, 'dataset', None) for dl in self.flattened]
        lengths = [length for ds in datasets if (length := sized_len(ds)) is not None]
        if not lengths:
            raise NotImplementedError('All datasets are iterable-style datasets.')
        fn = _SUPPORTED_MODES[self._mode]['fn']
        return fn(lengths)

    def _state_dicts(self) -> List[Dict[str, Any]]:
        """Returns the list of state dicts for iterables in `self.flattened` that are stateful."""
        return [loader.state_dict() for loader in self.flattened if isinstance(loader, _Stateful)]

    def _load_state_dicts(self, states: List[Dict[str, Any]]) -> None:
        """Loads the state dicts for iterables in `self.flattened` that are stateful."""
        if not states:
            return
        stateful_loaders = [loader for loader in self.flattened if isinstance(loader, _Stateful)]
        if len(stateful_loaders) != len(states):
            raise RuntimeError(f'The CombinedLoader has {len(stateful_loaders)} stateful loaders, but found {len(states)} states in the checkpoint. Please make sure you define the same dataloaders that were used when saving the checkpoint.')
        for loader, state_dict in zip(stateful_loaders, states):
            loader.load_state_dict(state_dict)