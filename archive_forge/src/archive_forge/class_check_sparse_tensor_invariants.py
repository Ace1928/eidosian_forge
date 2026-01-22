from typing import Optional, Tuple, List, Union, Any
import torch
from torch._C import _add_docstr, _sparse  # type: ignore[attr-defined]
from torch import Tensor
from .semi_structured import SparseSemiStructuredTensor, to_sparse_semi_structured
from typing import TYPE_CHECKING
class check_sparse_tensor_invariants:
    """A tool to control checking sparse tensor invariants.

    The following options exists to manage sparsr tensor invariants
    checking in sparse tensor construction:

    1. Using a context manager:

       .. code:: python

           with torch.sparse.check_sparse_tensor_invariants():
               run_my_model()

    2. Using a procedural approach:

       .. code:: python

           prev_checks_enabled = torch.sparse.check_sparse_tensor_invariants.is_enabled()
           torch.sparse.check_sparse_tensor_invariants.enable()

           run_my_model()

           if not prev_checks_enabled:
               torch.sparse.check_sparse_tensor_invariants.disable()

    3. Using function decoration:

       .. code:: python

           @torch.sparse.check_sparse_tensor_invariants()
           def run_my_model():
               ...

           run_my_model()

    4. Using ``check_invariants`` keyword argument in sparse tensor constructor call.
       For example:

       >>> torch.sparse_csr_tensor([0, 1, 3], [0, 1], [1, 2], check_invariants=True)
       Traceback (most recent call last):
         File "<stdin>", line 1, in <module>
       RuntimeError: `crow_indices[..., -1] == nnz` is not satisfied.
    """

    @staticmethod
    def is_enabled():
        """Return True if the sparse tensor invariants checking is enabled.

        .. note::

            Use :func:`torch.sparse.check_sparse_tensor_invariants.enable` or
            :func:`torch.sparse.check_sparse_tensor_invariants.disable` to
            manage the state of the sparse tensor invariants checks.
        """
        return torch._C._check_sparse_tensor_invariants()

    @staticmethod
    def enable():
        """Enable sparse tensor invariants checking in sparse tensor constructors.

        .. note::

            By default, the sparse tensor invariants checks are disabled. Use
            :func:`torch.sparse.check_sparse_tensor_invariants.is_enabled` to
            retrieve the current state of sparse tensor invariants checking.

        .. note::

            The sparse tensor invariants check flag is effective to all sparse
            tensor constructors, both in Python and ATen.

        The flag can be locally overridden by the ``check_invariants``
        optional argument of the sparse tensor constructor functions.
        """
        torch._C._set_check_sparse_tensor_invariants(True)

    @staticmethod
    def disable():
        """Disable sparse tensor invariants checking in sparse tensor constructors.

        See :func:`torch.sparse.check_sparse_tensor_invariants.enable` for more information.
        """
        torch._C._set_check_sparse_tensor_invariants(False)

    def __init__(self, enable=True):
        self.state = enable
        self.saved_state: Optional[bool] = None

    def __enter__(self):
        if self.saved_state is not None:
            raise RuntimeError('This context manager instance is already activated. Use a different context manager instance for context nesting.')
        self.saved_state = self.is_enabled()
        torch._C._set_check_sparse_tensor_invariants(self.state)

    def __exit__(self, type, value, traceback):
        assert self.saved_state is not None
        torch._C._set_check_sparse_tensor_invariants(self.saved_state)
        self.saved_state = None

    def __call__(self, mth):

        def test_mth(*args, **kwargs):
            with type(self)(self.state):
                return mth(*args, **kwargs)
        return test_mth