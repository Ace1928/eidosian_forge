from __future__ import annotations
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
import numpy as np
from xarray.backends.common import (
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import (
from xarray.core.variable import Variable
from xarray.namedarray.pycompat import integer_types
class PydapBackendEntrypoint(BackendEntrypoint):
    """
    Backend for steaming datasets over the internet using
    the Data Access Protocol, also known as DODS or OPeNDAP
    based on the pydap package.

    This backend is selected by default for urls.

    For more information about the underlying library, visit:
    https://www.pydap.org

    See Also
    --------
    backends.PydapDataStore
    """
    description = 'Open remote datasets via OPeNDAP using pydap in Xarray'
    url = 'https://docs.xarray.dev/en/stable/generated/xarray.backends.PydapBackendEntrypoint.html'

    def guess_can_open(self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore) -> bool:
        return isinstance(filename_or_obj, str) and is_remote_uri(filename_or_obj)

    def open_dataset(self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore, *, mask_and_scale=True, decode_times=True, concat_characters=True, decode_coords=True, drop_variables: str | Iterable[str] | None=None, use_cftime=None, decode_timedelta=None, application=None, session=None, output_grid=None, timeout=None, verify=None, user_charset=None) -> Dataset:
        store = PydapDataStore.open(url=filename_or_obj, application=application, session=session, output_grid=output_grid, timeout=timeout, verify=verify, user_charset=user_charset)
        store_entrypoint = StoreBackendEntrypoint()
        with close_on_error(store):
            ds = store_entrypoint.open_dataset(store, mask_and_scale=mask_and_scale, decode_times=decode_times, concat_characters=concat_characters, decode_coords=decode_coords, drop_variables=drop_variables, use_cftime=use_cftime, decode_timedelta=decode_timedelta)
            return ds