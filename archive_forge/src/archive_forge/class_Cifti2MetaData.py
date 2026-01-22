import re
from collections import OrderedDict
from collections.abc import Iterable, MutableMapping, MutableSequence
from warnings import warn
import numpy as np
from .. import xmlutils as xml
from ..arrayproxy import reshape_dataobj
from ..caret import CaretMetaData
from ..dataobj_images import DataobjImage
from ..filebasedimages import FileBasedHeader, SerializableImage
from ..nifti1 import Nifti1Extensions
from ..nifti2 import Nifti2Header, Nifti2Image
from ..volumeutils import Recoder, make_dt_codes
class Cifti2MetaData(CaretMetaData):
    """A list of name-value pairs

    * Description - Provides a simple method for user-supplied metadata that
      associates names with values.
    * Attributes: [NA]
    * Child Elements

        * MD (0...N)

    * Text Content: [NA]
    * Parent Elements - Matrix, NamedMap

    MD elements are a single metadata entry consisting of a name and a value.

    Attributes
    ----------
    data : list of (name, value) tuples
    """

    @staticmethod
    def _sanitize(args, kwargs):
        """Sanitize and warn on deprecated arguments

        Accept metadata positional/keyword argument that can take
        ``None`` to indicate no initialization.

        >>> import pytest
        >>> Cifti2MetaData()
        <Cifti2MetaData {}>
        >>> Cifti2MetaData([("key", "val")])
        <Cifti2MetaData {'key': 'val'}>
        >>> Cifti2MetaData(key="val")
        <Cifti2MetaData {'key': 'val'}>
        >>> with pytest.warns(FutureWarning):
        ...     Cifti2MetaData(None)
        <Cifti2MetaData {}>
        >>> with pytest.warns(FutureWarning):
        ...     Cifti2MetaData(metadata=None)
        <Cifti2MetaData {}>
        >>> with pytest.warns(FutureWarning):
        ...     Cifti2MetaData(metadata={'key': 'val'})
        <Cifti2MetaData {'key': 'val'}>

        Note that "metadata" could be a valid key:

        >>> Cifti2MetaData(metadata='val')
        <Cifti2MetaData {'metadata': 'val'}>
        """
        if not args and list(kwargs) == ['metadata']:
            if not isinstance(kwargs['metadata'], str):
                warn('Cifti2MetaData now has a dict-like interface and will no longer accept the ``metadata`` keyword argument in NiBabel 6.0. See ``pydoc dict`` for initialization options.', FutureWarning, stacklevel=3)
                md = kwargs.pop('metadata')
                if md is not None:
                    args = (md,)
        if args == (None,):
            warn('Cifti2MetaData now has a dict-like interface and will no longer accept the positional argument ``None`` in NiBabel 6.0. See ``pydoc dict`` for initialization options.', FutureWarning, stacklevel=3)
            args = ()
        return (args, kwargs)

    @property
    def data(self):
        return self._data

    def difference_update(self, metadata):
        """Remove metadata key-value pairs

        Parameters
        ----------
        metadata : dict-like datatype

        Returns
        -------
        None

        """
        if metadata is None:
            raise ValueError("The metadata parameter can't be None")
        pairs = dict(metadata)
        for k in pairs:
            del self.data[k]