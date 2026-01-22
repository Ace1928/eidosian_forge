from __future__ import annotations
import copy
import math
import sys
import warnings
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import (
import numpy as np
from xarray.core import dtypes, formatting, formatting_html
from xarray.core.indexing import (
from xarray.namedarray._aggregations import NamedArrayAggregations
from xarray.namedarray._typing import (
from xarray.namedarray.parallelcompat import guess_chunkmanager
from xarray.namedarray.pycompat import to_numpy
from xarray.namedarray.utils import (
class NamedArray(NamedArrayAggregations, Generic[_ShapeType_co, _DType_co]):
    """
    A wrapper around duck arrays with named dimensions
    and attributes which describe a single Array.
    Numeric operations on this object implement array broadcasting and
    dimension alignment based on dimension names,
    rather than axis order.


    Parameters
    ----------
    dims : str or iterable of hashable
        Name(s) of the dimension(s).
    data : array-like or duck-array
        The actual data that populates the array. Should match the
        shape specified by `dims`.
    attrs : dict, optional
        A dictionary containing any additional information or
        attributes you want to store with the array.
        Default is None, meaning no attributes will be stored.

    Raises
    ------
    ValueError
        If the `dims` length does not match the number of data dimensions (ndim).


    Examples
    --------
    >>> data = np.array([1.5, 2, 3], dtype=float)
    >>> narr = NamedArray(("x",), data, {"units": "m"})  # TODO: Better name than narr?
    """
    __slots__ = ('_data', '_dims', '_attrs')
    _data: duckarray[Any, _DType_co]
    _dims: _Dims
    _attrs: dict[Any, Any] | None

    def __init__(self, dims: _DimsLike, data: duckarray[Any, _DType_co], attrs: _AttrsLike=None):
        self._data = data
        self._dims = self._parse_dimensions(dims)
        self._attrs = dict(attrs) if attrs else None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        if NamedArray in cls.__bases__ and cls._new == NamedArray._new:
            raise TypeError('Subclasses of `NamedArray` must override the `_new` method.')
        super().__init_subclass__(**kwargs)

    @overload
    def _new(self, dims: _DimsLike | Default=..., data: duckarray[_ShapeType, _DType]=..., attrs: _AttrsLike | Default=...) -> NamedArray[_ShapeType, _DType]:
        ...

    @overload
    def _new(self, dims: _DimsLike | Default=..., data: Default=..., attrs: _AttrsLike | Default=...) -> NamedArray[_ShapeType_co, _DType_co]:
        ...

    def _new(self, dims: _DimsLike | Default=_default, data: duckarray[Any, _DType] | Default=_default, attrs: _AttrsLike | Default=_default) -> NamedArray[_ShapeType, _DType] | NamedArray[_ShapeType_co, _DType_co]:
        """
        Create a new array with new typing information.

        _new has to be reimplemented each time NamedArray is subclassed,
        otherwise type hints will not be correct. The same is likely true
        for methods that relied on _new.

        Parameters
        ----------
        dims : Iterable of Hashable, optional
            Name(s) of the dimension(s).
            Will copy the dims from x by default.
        data : duckarray, optional
            The actual data that populates the array. Should match the
            shape specified by `dims`.
            Will copy the data from x by default.
        attrs : dict, optional
            A dictionary containing any additional information or
            attributes you want to store with the array.
            Will copy the attrs from x by default.
        """
        return _new(self, dims, data, attrs)

    def _replace(self, dims: _DimsLike | Default=_default, data: duckarray[_ShapeType_co, _DType_co] | Default=_default, attrs: _AttrsLike | Default=_default) -> Self:
        """
        Create a new array with the same typing information.

        The types for each argument cannot change,
        use self._new if that is a risk.

        Parameters
        ----------
        dims : Iterable of Hashable, optional
            Name(s) of the dimension(s).
            Will copy the dims from x by default.
        data : duckarray, optional
            The actual data that populates the array. Should match the
            shape specified by `dims`.
            Will copy the data from x by default.
        attrs : dict, optional
            A dictionary containing any additional information or
            attributes you want to store with the array.
            Will copy the attrs from x by default.
        """
        return cast('Self', self._new(dims, data, attrs))

    def _copy(self, deep: bool=True, data: duckarray[_ShapeType_co, _DType_co] | None=None, memo: dict[int, Any] | None=None) -> Self:
        if data is None:
            ndata = self._data
            if deep:
                ndata = copy.deepcopy(ndata, memo=memo)
        else:
            ndata = data
            self._check_shape(ndata)
        attrs = copy.deepcopy(self._attrs, memo=memo) if deep else copy.copy(self._attrs)
        return self._replace(data=ndata, attrs=attrs)

    def __copy__(self) -> Self:
        return self._copy(deep=False)

    def __deepcopy__(self, memo: dict[int, Any] | None=None) -> Self:
        return self._copy(deep=True, memo=memo)

    def copy(self, deep: bool=True, data: duckarray[_ShapeType_co, _DType_co] | None=None) -> Self:
        """Returns a copy of this object.

        If `deep=True`, the data array is loaded into memory and copied onto
        the new object. Dimensions, attributes and encodings are always copied.

        Use `data` to create a new object with the same structure as
        original but entirely new data.

        Parameters
        ----------
        deep : bool, default: True
            Whether the data array is loaded into memory and copied onto
            the new object. Default is True.
        data : array_like, optional
            Data to use in the new object. Must have same shape as original.
            When `data` is used, `deep` is ignored.

        Returns
        -------
        object : NamedArray
            New object with dimensions, attributes, and optionally
            data copied from original.


        """
        return self._copy(deep=deep, data=data)

    @property
    def ndim(self) -> int:
        """
        Number of array dimensions.

        See Also
        --------
        numpy.ndarray.ndim
        """
        return len(self.shape)

    @property
    def size(self) -> _IntOrUnknown:
        """
        Number of elements in the array.

        Equal to ``np.prod(a.shape)``, i.e., the product of the array’s dimensions.

        See Also
        --------
        numpy.ndarray.size
        """
        return math.prod(self.shape)

    def __len__(self) -> _IntOrUnknown:
        try:
            return self.shape[0]
        except Exception as exc:
            raise TypeError('len() of unsized object') from exc

    @property
    def dtype(self) -> _DType_co:
        """
        Data-type of the array’s elements.

        See Also
        --------
        ndarray.dtype
        numpy.dtype
        """
        return self._data.dtype

    @property
    def shape(self) -> _Shape:
        """
        Get the shape of the array.

        Returns
        -------
        shape : tuple of ints
            Tuple of array dimensions.

        See Also
        --------
        numpy.ndarray.shape
        """
        return self._data.shape

    @property
    def nbytes(self) -> _IntOrUnknown:
        """
        Total bytes consumed by the elements of the data array.

        If the underlying data array does not include ``nbytes``, estimates
        the bytes consumed based on the ``size`` and ``dtype``.
        """
        if hasattr(self._data, 'nbytes'):
            return self._data.nbytes
        else:
            return self.size * self.dtype.itemsize

    @property
    def dims(self) -> _Dims:
        """Tuple of dimension names with which this NamedArray is associated."""
        return self._dims

    @dims.setter
    def dims(self, value: _DimsLike) -> None:
        self._dims = self._parse_dimensions(value)

    def _parse_dimensions(self, dims: _DimsLike) -> _Dims:
        dims = (dims,) if isinstance(dims, str) else tuple(dims)
        if len(dims) != self.ndim:
            raise ValueError(f'dimensions {dims} must have the same length as the number of data dimensions, ndim={self.ndim}')
        if len(set(dims)) < len(dims):
            repeated_dims = {d for d in dims if dims.count(d) > 1}
            warnings.warn(f"Duplicate dimension names present: dimensions {repeated_dims} appear more than once in dims={dims}. We do not yet support duplicate dimension names, but we do allow initial construction of the object. We recommend you rename the dims immediately to become distinct, as most xarray functionality is likely to fail silently if you do not. To rename the dimensions you will need to set the ``.dims`` attribute of each variable, ``e.g. var.dims=('x0', 'x1')``.", UserWarning)
        return dims

    @property
    def attrs(self) -> dict[Any, Any]:
        """Dictionary of local attributes on this NamedArray."""
        if self._attrs is None:
            self._attrs = {}
        return self._attrs

    @attrs.setter
    def attrs(self, value: Mapping[Any, Any]) -> None:
        self._attrs = dict(value) if value else None

    def _check_shape(self, new_data: duckarray[Any, _DType_co]) -> None:
        if new_data.shape != self.shape:
            raise ValueError(f"replacement data must match the {self.__class__.__name__}'s shape. replacement data has shape {new_data.shape}; {self.__class__.__name__} has shape {self.shape}")

    @property
    def data(self) -> duckarray[Any, _DType_co]:
        """
        The NamedArray's data as an array. The underlying array type
        (e.g. dask, sparse, pint) is preserved.

        """
        return self._data

    @data.setter
    def data(self, data: duckarray[Any, _DType_co]) -> None:
        self._check_shape(data)
        self._data = data

    @property
    def imag(self: NamedArray[_ShapeType, np.dtype[_SupportsImag[_ScalarType]]]) -> NamedArray[_ShapeType, _dtype[_ScalarType]]:
        """
        The imaginary part of the array.

        See Also
        --------
        numpy.ndarray.imag
        """
        if isinstance(self._data, _arrayapi):
            from xarray.namedarray._array_api import imag
            return imag(self)
        return self._new(data=self._data.imag)

    @property
    def real(self: NamedArray[_ShapeType, np.dtype[_SupportsReal[_ScalarType]]]) -> NamedArray[_ShapeType, _dtype[_ScalarType]]:
        """
        The real part of the array.

        See Also
        --------
        numpy.ndarray.real
        """
        if isinstance(self._data, _arrayapi):
            from xarray.namedarray._array_api import real
            return real(self)
        return self._new(data=self._data.real)

    def __dask_tokenize__(self) -> object:
        from dask.base import normalize_token
        return normalize_token((type(self), self._dims, self.data, self._attrs or None))

    def __dask_graph__(self) -> Graph | None:
        if is_duck_dask_array(self._data):
            return self._data.__dask_graph__()
        else:
            return None

    def __dask_keys__(self) -> NestedKeys:
        if is_duck_dask_array(self._data):
            return self._data.__dask_keys__()
        else:
            raise AttributeError('Method requires self.data to be a dask array.')

    def __dask_layers__(self) -> Sequence[str]:
        if is_duck_dask_array(self._data):
            return self._data.__dask_layers__()
        else:
            raise AttributeError('Method requires self.data to be a dask array.')

    @property
    def __dask_optimize__(self) -> Callable[..., dict[Any, Any]]:
        if is_duck_dask_array(self._data):
            return self._data.__dask_optimize__
        else:
            raise AttributeError('Method requires self.data to be a dask array.')

    @property
    def __dask_scheduler__(self) -> SchedulerGetCallable:
        if is_duck_dask_array(self._data):
            return self._data.__dask_scheduler__
        else:
            raise AttributeError('Method requires self.data to be a dask array.')

    def __dask_postcompute__(self) -> tuple[PostComputeCallable, tuple[Any, ...]]:
        if is_duck_dask_array(self._data):
            array_func, array_args = self._data.__dask_postcompute__()
            return (self._dask_finalize, (array_func,) + array_args)
        else:
            raise AttributeError('Method requires self.data to be a dask array.')

    def __dask_postpersist__(self) -> tuple[Callable[[Graph, PostPersistCallable[Any], Any, Any], Self], tuple[Any, ...]]:
        if is_duck_dask_array(self._data):
            a: tuple[PostPersistCallable[Any], tuple[Any, ...]]
            a = self._data.__dask_postpersist__()
            array_func, array_args = a
            return (self._dask_finalize, (array_func,) + array_args)
        else:
            raise AttributeError('Method requires self.data to be a dask array.')

    def _dask_finalize(self, results: Graph, array_func: PostPersistCallable[Any], *args: Any, **kwargs: Any) -> Self:
        data = array_func(results, *args, **kwargs)
        return type(self)(self._dims, data, attrs=self._attrs)

    @overload
    def get_axis_num(self, dim: Iterable[Hashable]) -> tuple[int, ...]:
        ...

    @overload
    def get_axis_num(self, dim: Hashable) -> int:
        ...

    def get_axis_num(self, dim: Hashable | Iterable[Hashable]) -> int | tuple[int, ...]:
        """Return axis number(s) corresponding to dimension(s) in this array.

        Parameters
        ----------
        dim : str or iterable of str
            Dimension name(s) for which to lookup axes.

        Returns
        -------
        int or tuple of int
            Axis number or numbers corresponding to the given dimensions.
        """
        if not isinstance(dim, str) and isinstance(dim, Iterable):
            return tuple((self._get_axis_num(d) for d in dim))
        else:
            return self._get_axis_num(dim)

    def _get_axis_num(self: Any, dim: Hashable) -> int:
        _raise_if_any_duplicate_dimensions(self.dims)
        try:
            return self.dims.index(dim)
        except ValueError:
            raise ValueError(f'{dim!r} not found in array dimensions {self.dims!r}')

    @property
    def chunks(self) -> _Chunks | None:
        """
        Tuple of block lengths for this NamedArray's data, in order of dimensions, or None if
        the underlying data is not a dask array.

        See Also
        --------
        NamedArray.chunk
        NamedArray.chunksizes
        xarray.unify_chunks
        """
        data = self._data
        if isinstance(data, _chunkedarray):
            return data.chunks
        else:
            return None

    @property
    def chunksizes(self) -> Mapping[_Dim, _Shape]:
        """
        Mapping from dimension names to block lengths for this namedArray's data, or None if
        the underlying data is not a dask array.
        Cannot be modified directly, but can be modified by calling .chunk().

        Differs from NamedArray.chunks because it returns a mapping of dimensions to chunk shapes
        instead of a tuple of chunk shapes.

        See Also
        --------
        NamedArray.chunk
        NamedArray.chunks
        xarray.unify_chunks
        """
        data = self._data
        if isinstance(data, _chunkedarray):
            return dict(zip(self.dims, data.chunks))
        else:
            return {}

    @property
    def sizes(self) -> dict[_Dim, _IntOrUnknown]:
        """Ordered mapping from dimension names to lengths."""
        return dict(zip(self.dims, self.shape))

    def chunk(self, chunks: int | Literal['auto'] | Mapping[Any, None | int | tuple[int, ...]]={}, chunked_array_type: str | ChunkManagerEntrypoint[Any] | None=None, from_array_kwargs: Any=None, **chunks_kwargs: Any) -> Self:
        """Coerce this array's data into a dask array with the given chunks.

        If this variable is a non-dask array, it will be converted to dask
        array. If it's a dask array, it will be rechunked to the given chunk
        sizes.

        If neither chunks is not provided for one or more dimensions, chunk
        sizes along that dimension will not be updated; non-dask arrays will be
        converted into dask arrays with a single block.

        Parameters
        ----------
        chunks : int, tuple or dict, optional
            Chunk sizes along each dimension, e.g., ``5``, ``(5, 5)`` or
            ``{'x': 5, 'y': 5}``.
        chunked_array_type: str, optional
            Which chunked array type to coerce this datasets' arrays to.
            Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEntrypoint` system.
            Experimental API that should not be relied upon.
        from_array_kwargs: dict, optional
            Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
            chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
            For example, with dask as the default chunked array type, this method would pass additional kwargs
            to :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.
        **chunks_kwargs : {dim: chunks, ...}, optional
            The keyword arguments form of ``chunks``.
            One of chunks or chunks_kwargs must be provided.

        Returns
        -------
        chunked : xarray.Variable

        See Also
        --------
        Variable.chunks
        Variable.chunksizes
        xarray.unify_chunks
        dask.array.from_array
        """
        if from_array_kwargs is None:
            from_array_kwargs = {}
        if chunks is None:
            warnings.warn("None value for 'chunks' is deprecated. It will raise an error in the future. Use instead '{}'", category=FutureWarning)
            chunks = {}
        if isinstance(chunks, (float, str, int, tuple, list)):
            pass
        else:
            chunks = either_dict_or_kwargs(chunks, chunks_kwargs, 'chunk')
        if is_dict_like(chunks):
            chunks = {self.get_axis_num(dim): chunk for dim, chunk in chunks.items()}
        chunkmanager = guess_chunkmanager(chunked_array_type)
        data_old = self._data
        if chunkmanager.is_chunked_array(data_old):
            data_chunked = chunkmanager.rechunk(data_old, chunks)
        else:
            if not isinstance(data_old, ExplicitlyIndexed):
                ndata = data_old
            else:
                ndata = ImplicitToExplicitIndexingAdapter(data_old, OuterIndexer)
            if is_dict_like(chunks):
                chunks = tuple((chunks.get(n, s) for n, s in enumerate(ndata.shape)))
            data_chunked = chunkmanager.from_array(ndata, chunks, **from_array_kwargs)
        return self._replace(data=data_chunked)

    def to_numpy(self) -> np.ndarray[Any, Any]:
        """Coerces wrapped data to numpy and returns a numpy.ndarray"""
        return to_numpy(self._data)

    def as_numpy(self) -> Self:
        """Coerces wrapped data into a numpy array, returning a Variable."""
        return self._replace(data=self.to_numpy())

    def reduce(self, func: Callable[..., Any], dim: Dims=None, axis: int | Sequence[int] | None=None, keepdims: bool=False, **kwargs: Any) -> NamedArray[Any, Any]:
        """Reduce this array by applying `func` along some dimension(s).

        Parameters
        ----------
        func : callable
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of reducing an
            np.ndarray over an integer valued axis.
        dim : "...", str, Iterable of Hashable or None, optional
            Dimension(s) over which to apply `func`. By default `func` is
            applied over all dimensions.
        axis : int or Sequence of int, optional
            Axis(es) over which to apply `func`. Only one of the 'dim'
            and 'axis' arguments can be supplied. If neither are supplied, then
            the reduction is calculated over the flattened array (by calling
            `func(x)` without an axis argument).
        keepdims : bool, default: False
            If True, the dimensions which are reduced are left in the result
            as dimensions of size one
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Array
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        if dim == ...:
            dim = None
        if dim is not None and axis is not None:
            raise ValueError("cannot supply both 'axis' and 'dim' arguments")
        if dim is not None:
            axis = self.get_axis_num(dim)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Mean of empty slice', category=RuntimeWarning)
            if axis is not None:
                if isinstance(axis, tuple) and len(axis) == 1:
                    axis = axis[0]
                data = func(self.data, axis=axis, **kwargs)
            else:
                data = func(self.data, **kwargs)
        if getattr(data, 'shape', ()) == self.shape:
            dims = self.dims
        else:
            removed_axes: Iterable[int]
            if axis is None:
                removed_axes = range(self.ndim)
            else:
                removed_axes = np.atleast_1d(axis) % self.ndim
            if keepdims:
                slices = tuple((np.newaxis if i in removed_axes else slice(None, None) for i in range(self.ndim)))
                if getattr(data, 'shape', None) is None:
                    data = np.asanyarray(data)[slices]
                else:
                    data = data[slices]
                dims = self.dims
            else:
                dims = tuple((adim for n, adim in enumerate(self.dims) if n not in removed_axes))
        return from_array(dims, data, attrs=self._attrs)

    def _nonzero(self: T_NamedArrayInteger) -> tuple[T_NamedArrayInteger, ...]:
        """Equivalent numpy's nonzero but returns a tuple of NamedArrays."""
        nonzeros = np.nonzero(cast('NDArray[np.integer[Any]]', self.data))
        _attrs = self.attrs
        return tuple((cast('T_NamedArrayInteger', self._new((dim,), nz, _attrs)) for nz, dim in zip(nonzeros, self.dims)))

    def __repr__(self) -> str:
        return formatting.array_repr(self)

    def _repr_html_(self) -> str:
        return formatting_html.array_repr(self)

    def _as_sparse(self, sparse_format: Literal['coo'] | Default=_default, fill_value: ArrayLike | Default=_default) -> NamedArray[Any, _DType_co]:
        """
        Use sparse-array as backend.
        """
        import sparse
        from xarray.namedarray._array_api import astype
        if fill_value is _default:
            dtype, fill_value = dtypes.maybe_promote(self.dtype)
        else:
            dtype = dtypes.result_type(self.dtype, fill_value)
        if sparse_format is _default:
            sparse_format = 'coo'
        try:
            as_sparse = getattr(sparse, f'as_{sparse_format.lower()}')
        except AttributeError as exc:
            raise ValueError(f'{sparse_format} is not a valid sparse format') from exc
        data = as_sparse(astype(self, dtype).data, fill_value=fill_value)
        return self._new(data=data)

    def _to_dense(self) -> NamedArray[Any, _DType_co]:
        """
        Change backend from sparse to np.array.
        """
        if isinstance(self._data, _sparsearrayfunction_or_api):
            data_dense: np.ndarray[Any, _DType_co] = self._data.todense()
            return self._new(data=data_dense)
        else:
            raise TypeError('self.data is not a sparse array')

    def permute_dims(self, *dim: Iterable[_Dim] | ellipsis, missing_dims: ErrorOptionsWithWarn='raise') -> NamedArray[Any, _DType_co]:
        """Return a new object with transposed dimensions.

        Parameters
        ----------
        *dim : Hashable, optional
            By default, reverse the order of the dimensions. Otherwise, reorder the
            dimensions to this order.
        missing_dims : {"raise", "warn", "ignore"}, default: "raise"
            What to do if dimensions that should be selected from are not present in the
            NamedArray:
            - "raise": raise an exception
            - "warn": raise a warning, and ignore the missing dimensions
            - "ignore": ignore the missing dimensions

        Returns
        -------
        NamedArray
            The returned NamedArray has permuted dimensions and data with the
            same attributes as the original.


        See Also
        --------
        numpy.transpose
        """
        from xarray.namedarray._array_api import permute_dims
        if not dim:
            dims = self.dims[::-1]
        else:
            dims = tuple(infix_dims(dim, self.dims, missing_dims))
        if len(dims) < 2 or dims == self.dims:
            return self.copy(deep=False)
        axes_result = self.get_axis_num(dims)
        axes = (axes_result,) if isinstance(axes_result, int) else axes_result
        return permute_dims(self, axes)

    @property
    def T(self) -> NamedArray[Any, _DType_co]:
        """Return a new object with transposed dimensions."""
        if self.ndim != 2:
            raise ValueError(f'x.T requires x to have 2 dimensions, got {self.ndim}. Use x.permute_dims() to permute dimensions.')
        return self.permute_dims()

    def broadcast_to(self, dim: Mapping[_Dim, int] | None=None, **dim_kwargs: Any) -> NamedArray[Any, _DType_co]:
        """
        Broadcast the NamedArray to a new shape. New dimensions are not allowed.

        This method allows for the expansion of the array's dimensions to a specified shape.
        It handles both positional and keyword arguments for specifying the dimensions to broadcast.
        An error is raised if new dimensions are attempted to be added.

        Parameters
        ----------
        dim : dict, str, sequence of str, optional
            Dimensions to broadcast the array to. If a dict, keys are dimension names and values are the new sizes.
            If a string or sequence of strings, existing dimensions are matched with a size of 1.

        **dim_kwargs : Any
            Additional dimensions specified as keyword arguments. Each keyword argument specifies the name of an existing dimension and its size.

        Returns
        -------
        NamedArray
            A new NamedArray with the broadcasted dimensions.

        Examples
        --------
        >>> data = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        >>> array = xr.NamedArray(("x", "y"), data)
        >>> array.sizes
        {'x': 2, 'y': 2}

        >>> broadcasted = array.broadcast_to(x=2, y=2)
        >>> broadcasted.sizes
        {'x': 2, 'y': 2}
        """
        from xarray.core import duck_array_ops
        combined_dims = either_dict_or_kwargs(dim, dim_kwargs, 'broadcast_to')
        if (new_dims := (set(combined_dims) - set(self.dims))):
            raise ValueError(f'Cannot add new dimensions: {new_dims}. Only existing dimensions are allowed. Use `expand_dims` method to add new dimensions.')
        current_shape = self.sizes
        broadcast_shape = {d: current_shape.get(d, 1) for d in self.dims}
        broadcast_shape |= combined_dims
        ordered_dims = list(broadcast_shape.keys())
        ordered_shape = tuple((broadcast_shape[d] for d in ordered_dims))
        data = duck_array_ops.broadcast_to(self._data, ordered_shape)
        return self._new(data=data, dims=ordered_dims)

    def expand_dims(self, dim: _Dim | Default=_default) -> NamedArray[Any, _DType_co]:
        """
        Expand the dimensions of the NamedArray.

        This method adds new dimensions to the object. The new dimensions are added at the beginning of the array.

        Parameters
        ----------
        dim : Hashable, optional
            Dimension name to expand the array to. This dimension will be added at the beginning of the array.

        Returns
        -------
        NamedArray
            A new NamedArray with expanded dimensions.


        Examples
        --------

        >>> data = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        >>> array = xr.NamedArray(("x", "y"), data)


        # expand dimensions by specifying a new dimension name
        >>> expanded = array.expand_dims(dim="z")
        >>> expanded.dims
        ('z', 'x', 'y')

        """
        from xarray.namedarray._array_api import expand_dims
        return expand_dims(self, dim=dim)