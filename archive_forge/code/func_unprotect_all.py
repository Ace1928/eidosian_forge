import contextlib
import typing
from . import openrlib
from _cffi_backend import FFI  # type: ignore
def unprotect_all(self) -> None:
    """Release the total count of objects this instance knows to be
        protected from the protection stack."""
    openrlib.rlib.Rf_unprotect(self._counter)
    self._counter = 0