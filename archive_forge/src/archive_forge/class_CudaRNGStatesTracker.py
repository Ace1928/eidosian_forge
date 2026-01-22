import contextlib
from typing import Dict, Iterator, Set, Union
import torch
from torch.cuda import _lazy_call
from torch.utils.checkpoint import detach_variable
from .initialize import get_data_parallel_rank, get_model_parallel_rank
class CudaRNGStatesTracker:
    """Tracker for the cuda RNG states.

    Using the `add` method, a cuda rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    cuda state.
    """

    def __init__(self) -> None:
        self.states_: Dict[str, torch.ByteTensor] = {}
        self.seeds_: Set[int] = set()

    def reset(self) -> None:
        """Set to the initial state (no tracker)."""
        self.states_ = {}
        self.seeds_ = set()

    def get_states(self) -> Dict[str, torch.ByteTensor]:
        """Get rng states. Copy the dictionary so we have direct
        pointers to the states, not just a pointer to the dictionary."""
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states(self, states: Dict[str, torch.ByteTensor]) -> None:
        """Set the rng states. For efficiency purposes, we do not check
        the size of seed for compatibility."""
        self.states_ = states

    def add(self, name: str, seed: int) -> None:
        """Track the rng state.
        Arguments:
            name (str): The name of the seed
            seed (int): The seed value
        """
        if seed in self.seeds_:
            raise Exception('seed {} already exists'.format(seed))
        self.seeds_.add(seed)
        if name in self.states_:
            raise Exception('cuda rng state {} already exists'.format(name))
        orig_rng_state = torch.cuda.get_rng_state()
        torch.cuda.manual_seed(seed)
        self.states_[name] = torch.cuda.get_rng_state()
        _set_cuda_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def fork(self, name: str=_MODEL_PARALLEL_RNG_TRACKER_NAME) -> Iterator[None]:
        """Fork the cuda rng state, perform operations, and exit with
        the original state."""
        if name not in self.states_:
            raise Exception('cuda rng state {} is not added'.format(name))
        orig_cuda_rng_state = torch.cuda.get_rng_state()
        _set_cuda_rng_state(self.states_[name])
        try:
            yield
        finally:
            self.states_[name] = torch.cuda.get_rng_state()
            _set_cuda_rng_state(orig_cuda_rng_state)