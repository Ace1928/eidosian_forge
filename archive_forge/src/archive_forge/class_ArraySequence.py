import numbers
from functools import reduce
from operator import mul
import numpy as np
@_define_operators
class ArraySequence:
    """Sequence of ndarrays having variable first dimension sizes.

    This is a container that can store multiple ndarrays where each ndarray
    might have a different first dimension size but a *common* size for the
    remaining dimensions.

    More generally, an instance of :class:`ArraySequence` of length $N$ is
    composed of $N$ ndarrays of shape $(d_1, d_2, ... d_D)$ where $d_1$
    can vary in length between arrays but $(d_2, ..., d_D)$ have to be the
    same for every ndarray.
    """

    def __init__(self, iterable=None, buffer_size=4):
        """Initialize array sequence instance

        Parameters
        ----------
        iterable : None or iterable or :class:`ArraySequence`, optional
            If None, create an empty :class:`ArraySequence` object.
            If iterable, create a :class:`ArraySequence` object initialized
            from array-like objects yielded by the iterable.
            If :class:`ArraySequence`, create a view (no memory is allocated).
            For an actual copy use :meth:`.copy` instead.
        buffer_size : float, optional
            Size (in Mb) for memory allocation when `iterable` is a generator.
        """
        self._is_view = False
        self._data = np.array([])
        self._offsets = np.array([], dtype=np.intp)
        self._lengths = np.array([], dtype=np.intp)
        self._buffer_size = buffer_size
        self._build_cache = None
        if iterable is None:
            return
        if is_array_sequence(iterable):
            self._data = iterable._data
            self._offsets = iterable._offsets
            self._lengths = iterable._lengths
            self._is_view = True
            return
        self.extend(iterable)

    @property
    def is_sliced_view(self):
        return self._lengths.sum() != self._data.shape[0]

    @property
    def is_array_sequence(self):
        return True

    @property
    def common_shape(self):
        """Matching shape of the elements in this array sequence."""
        return self._data.shape[1:]

    @property
    def total_nb_rows(self):
        """Total number of rows in this array sequence."""
        return np.sum(self._lengths)

    def get_data(self):
        """Returns a *copy* of the elements in this array sequence.

        Notes
        -----
        To modify the data on this array sequence, one can use
        in-place mathematical operators (e.g., `seq += ...`) or the use
        assignment operator (i.e, `seq[...] = value`).
        """
        return self.copy()._data

    def _check_shape(self, arrseq):
        """Check whether this array sequence is compatible with another."""
        msg = 'cannot perform operation - array sequences have different'
        if len(self._lengths) != len(arrseq._lengths):
            msg += f' lengths: {len(self._lengths)} vs. {len(arrseq._lengths)}.'
            raise ValueError(msg)
        if self.total_nb_rows != arrseq.total_nb_rows:
            msg += f' amount of data: {self.total_nb_rows} vs. {arrseq.total_nb_rows}.'
            raise ValueError(msg)
        if self.common_shape != arrseq.common_shape:
            msg += f' common shape: {self.common_shape} vs. {arrseq.common_shape}.'
            raise ValueError(msg)
        return True

    def _get_next_offset(self):
        """Offset in ``self._data`` at which to write next rowelement"""
        if len(self._offsets) == 0:
            return 0
        imax = np.argmax(self._offsets)
        return self._offsets[imax] + self._lengths[imax]

    def append(self, element, cache_build=False):
        """Appends `element` to this array sequence.

        Append can be a lot faster if it knows that it is appending several
        elements instead of a single element.  In that case it can cache the
        parameters it uses between append operations, in a "build cache".  To
        tell append to do this, use ``cache_build=True``.  If you use
        ``cache_build=True``, you need to finalize the append operations with
        :meth:`finalize_append`.

        Parameters
        ----------
        element : ndarray
            Element to append. The shape must match already inserted elements
            shape except for the first dimension.
        cache_build : {False, True}
            Whether to save the build cache from this append routine.  If True,
            append can assume it is the only player updating `self`, and the
            caller must finalize `self` after all append operations, with
            ``self.finalize_append()``.

        Returns
        -------
        None

        Notes
        -----
        If you need to add multiple elements you should consider
        `ArraySequence.extend`.
        """
        element = np.asarray(element)
        if element.size == 0:
            return
        el_shape = element.shape
        n_items, common_shape = (el_shape[0], el_shape[1:])
        build_cache = self._build_cache
        in_cached_build = build_cache is not None
        if not in_cached_build:
            build_cache = _BuildCache(self, common_shape, element.dtype)
        next_offset = build_cache.next_offset
        req_rows = next_offset + n_items
        if self._data.shape[0] < req_rows:
            self._resize_data_to(req_rows, build_cache)
        self._data[next_offset:req_rows] = element
        build_cache.offsets.append(next_offset)
        build_cache.lengths.append(n_items)
        build_cache.next_offset = req_rows
        if in_cached_build:
            return
        if cache_build:
            self._build_cache = build_cache
        else:
            build_cache.update_seq(self)

    def finalize_append(self):
        """Finalize process of appending several elements to `self`

        :meth:`append` can be a lot faster if it knows that it is appending
        several elements instead of a single element.  To tell the append
        method this is the case, use ``cache_build=True``.  This method
        finalizes the series of append operations after a call to
        :meth:`append` with ``cache_build=True``.
        """
        if self._build_cache is None:
            return
        self._build_cache.update_seq(self)
        self._build_cache = None
        self.shrink_data()

    def _resize_data_to(self, n_rows, build_cache):
        """Resize data array if required"""
        n_bufs = np.ceil(n_rows / build_cache.rows_per_buf)
        extended_n_rows = int(n_bufs * build_cache.rows_per_buf)
        new_shape = (extended_n_rows,) + build_cache.common_shape
        if self._data.size == 0:
            self._data = np.empty(new_shape, dtype=build_cache.dtype)
        else:
            try:
                self._data.resize(new_shape)
            except ValueError:
                self._data = self._data.copy()
                self._data.resize(new_shape, refcheck=False)

    def shrink_data(self):
        self._data.resize((self._get_next_offset(),) + self.common_shape, refcheck=False)

    def extend(self, elements):
        """Appends all `elements` to this array sequence.

        Parameters
        ----------
        elements : iterable of ndarrays or :class:`ArraySequence` object
            If iterable of ndarrays, each ndarray will be concatenated along
            the first dimension then appended to the data of this
            ArraySequence.
            If :class:`ArraySequence` object, its data are simply appended to
            the data of this ArraySequence.

        Returns
        -------
        None

        Notes
        -----
        The shape of the elements to be added must match the one of the data of
        this :class:`ArraySequence` except for the first dimension.
        """
        try:
            iter_len = len(elements)
        except TypeError:
            pass
        else:
            if iter_len == 0:
                return
            e0 = np.asarray(elements[0])
            n_elements = np.sum([len(e) for e in elements])
            self._build_cache = _BuildCache(self, e0.shape[1:], e0.dtype)
            self._resize_data_to(self._get_next_offset() + n_elements, self._build_cache)
        for e in elements:
            self.append(e, cache_build=True)
        self.finalize_append()

    def copy(self):
        """Creates a copy of this :class:`ArraySequence` object.

        Returns
        -------
        seq_copy : :class:`ArraySequence` instance
            Copy of `self`.

        Notes
        -----
        We do not simply deepcopy this object because we have a chance to use
        less memory. For example, if the array sequence being copied is the
        result of a slicing operation on an array sequence.
        """
        seq = self.__class__()
        total_lengths = np.sum(self._lengths)
        seq._data = np.empty((total_lengths,) + self._data.shape[1:], dtype=self._data.dtype)
        next_offset = 0
        offsets = []
        for offset, length in zip(self._offsets, self._lengths):
            offsets.append(next_offset)
            chunk = self._data[offset:offset + length]
            seq._data[next_offset:next_offset + length] = chunk
            next_offset += length
        seq._offsets = np.asarray(offsets)
        seq._lengths = self._lengths.copy()
        return seq

    def __getitem__(self, idx):
        """Get sequence(s) through standard or advanced numpy indexing.

        Parameters
        ----------
        idx : int or slice or list or ndarray
            If int, index of the element to retrieve.
            If slice, use slicing to retrieve elements.
            If list, indices of the elements to retrieve.
            If ndarray with dtype int, indices of the elements to retrieve.
            If ndarray with dtype bool, only retrieve selected elements.

        Returns
        -------
        ndarray or :class:`ArraySequence`
            If `idx` is an int, returns the selected sequence.
            Otherwise, returns a :class:`ArraySequence` object which is a view
            of the selected sequences.
        """
        if isinstance(idx, (numbers.Integral, np.integer)):
            start = self._offsets[idx]
            return self._data[start:start + self._lengths[idx]]
        seq = self.__class__()
        seq._is_view = True
        if isinstance(idx, tuple):
            off_idx = idx[0]
            seq._data = self._data.__getitem__((slice(None),) + idx[1:])
        else:
            off_idx = idx
            seq._data = self._data
        if isinstance(off_idx, slice):
            seq._offsets = self._offsets[off_idx]
            seq._lengths = self._lengths[off_idx]
            return seq
        if isinstance(off_idx, (list, range)) or is_ndarray_of_int_or_bool(off_idx):
            seq._offsets = self._offsets[off_idx]
            seq._lengths = self._lengths[off_idx]
            return seq
        raise TypeError('Index must be either an int, a slice, a list of int or a ndarray of bool! Not ' + str(type(idx)))

    def __setitem__(self, idx, elements):
        """Set sequence(s) through standard or advanced numpy indexing.

        Parameters
        ----------
        idx : int or slice or list or ndarray
            If int, index of the element to retrieve.
            If slice, use slicing to retrieve elements.
            If list, indices of the elements to retrieve.
            If ndarray with dtype int, indices of the elements to retrieve.
            If ndarray with dtype bool, only retrieve selected elements.
        elements: ndarray or :class:`ArraySequence`
            Data that will overwrite selected sequences.
            If `idx` is an int, `elements` is expected to be a ndarray.
            Otherwise, `elements` is expected a :class:`ArraySequence` object.
        """
        if isinstance(idx, (numbers.Integral, np.integer)):
            start = self._offsets[idx]
            self._data[start:start + self._lengths[idx]] = elements
            return
        if isinstance(idx, tuple):
            off_idx = idx[0]
            data = self._data.__getitem__((slice(None),) + idx[1:])
        else:
            off_idx = idx
            data = self._data
        if isinstance(off_idx, slice):
            offsets = self._offsets[off_idx]
            lengths = self._lengths[off_idx]
        elif isinstance(off_idx, (list, range)) or is_ndarray_of_int_or_bool(off_idx):
            offsets = self._offsets[off_idx]
            lengths = self._lengths[off_idx]
        else:
            raise TypeError('Index must be either an int, a slice, a list of int or a ndarray of bool! Not ' + str(type(idx)))
        if is_array_sequence(elements):
            if len(lengths) != len(elements):
                msg = f'Trying to set {len(lengths)} sequences with {len(elements)} sequences.'
                raise ValueError(msg)
            if sum(lengths) != elements.total_nb_rows:
                msg = f'Trying to set {sum(lengths)} points with {elements.total_nb_rows} points.'
                raise ValueError(msg)
            for o1, l1, o2, l2 in zip(offsets, lengths, elements._offsets, elements._lengths):
                data[o1:o1 + l1] = elements._data[o2:o2 + l2]
        elif isinstance(elements, numbers.Number):
            for o1, l1 in zip(offsets, lengths):
                data[o1:o1 + l1] = elements
        else:
            for o1, l1, element in zip(offsets, lengths, elements):
                data[o1:o1 + l1] = element

    def _op(self, op, value=None, inplace=False):
        """Applies some operator to this arraysequence.

        This handles both unary and binary operators with a scalar or another
        array sequence. Operations are performed directly on the underlying
        data, or a copy of it, which depends on the value of `inplace`.

        Parameters
        ----------
        op : str
            Name of the Python operator (e.g., `"__add__"`).
        value : scalar or :class:`ArraySequence`, optional
            If None, the operator is assumed to be unary.
            Otherwise, that value is used in the binary operation.
        inplace: bool, optional
            If False, the operation is done on a copy of this array sequence.
            Otherwise, this array sequence gets modified directly.
        """
        seq = self if inplace else self.copy()
        if is_array_sequence(value) and seq._check_shape(value):
            elements = zip(seq._offsets, seq._lengths, self._offsets, self._lengths, value._offsets, value._lengths)
            o0, l0, o1, l1, o2, l2 = next(elements)
            tmp = getattr(self._data[o1:o1 + l1], op)(value._data[o2:o2 + l2])
            seq._data = seq._data.astype(tmp.dtype)
            seq._data[o0:o0 + l0] = tmp
            for o0, l0, o1, l1, o2, l2 in elements:
                seq._data[o0:o0 + l0] = getattr(self._data[o1:o1 + l1], op)(value._data[o2:o2 + l2])
        else:
            args = [] if value is None else [value]
            elements = zip(seq._offsets, seq._lengths, self._offsets, self._lengths)
            o0, l0, o1, l1 = next(elements)
            tmp = getattr(self._data[o1:o1 + l1], op)(*args)
            seq._data = seq._data.astype(tmp.dtype)
            seq._data[o0:o0 + l0] = tmp
            for o0, l0, o1, l1 in elements:
                seq._data[o0:o0 + l0] = getattr(self._data[o1:o1 + l1], op)(*args)
        return seq

    def __iter__(self):
        if len(self._lengths) != len(self._offsets):
            raise ValueError('ArraySequence object corrupted: len(self._lengths) != len(self._offsets)')
        for offset, lengths in zip(self._offsets, self._lengths):
            yield self._data[offset:offset + lengths]

    def __len__(self):
        return len(self._offsets)

    def __repr__(self):
        if len(self) > np.get_printoptions()['threshold']:
            edgeitems = np.get_printoptions()['edgeitems']
            data = str(list(self[:edgeitems]))[:-1]
            data += ', ..., '
            data += str(list(self[-edgeitems:]))[1:]
        else:
            data = str(list(self))
        return f'{self.__class__.__name__}({data})'

    def save(self, filename):
        """Saves this :class:`ArraySequence` object to a .npz file."""
        np.savez(filename, data=self._data, offsets=self._offsets, lengths=self._lengths)

    @classmethod
    def load(cls, filename):
        """Loads a :class:`ArraySequence` object from a .npz file."""
        content = np.load(filename)
        seq = cls()
        seq._data = content['data']
        seq._offsets = content['offsets']
        seq._lengths = content['lengths']
        return seq