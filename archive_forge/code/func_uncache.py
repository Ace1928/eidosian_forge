from __future__ import annotations
import typing as ty
import numpy as np
from .arrayproxy import ArrayLike
from .deprecated import deprecate_with_version
from .filebasedimages import FileBasedHeader, FileBasedImage
from .fileholders import FileMap
def uncache(self) -> None:
    """Delete any cached read of data from proxied data

        Remember there are two types of images:

        * *array images* where the data ``img.dataobj`` is an array
        * *proxy images* where the data ``img.dataobj`` is a proxy object

        If you call ``img.get_fdata()`` on a proxy image, the result of reading
        from the proxy gets cached inside the image object, and this cache is
        what gets returned from the next call to ``img.get_fdata()``.  If you
        modify the returned data, as in::

            data = img.get_fdata()
            data[:] = 42

        then the next call to ``img.get_fdata()`` returns the modified array,
        whether the image is an array image or a proxy image::

            assert np.all(img.get_fdata() == 42)

        When you uncache an array image, this has no effect on the return of
        ``img.get_fdata()``, but when you uncache a proxy image, the result of
        ``img.get_fdata()`` returns to its original value.
        """
    self._fdata_cache = None
    self._data_cache = None