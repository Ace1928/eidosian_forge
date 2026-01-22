import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
class Ops:
    name: str = 'base'
    xp: Xp = numpy

    def __init__(self, device_type: DeviceTypes='cpu', device_id: int=-1, **kwargs) -> None:
        self.device_type = device_type
        self.device_id = device_id

    def cblas(self) -> CBlas:
        """Return C BLAS function table."""
        err = f'{type(self).__name__} does not provide C BLAS functions'
        raise NotImplementedError(err)

    def to_numpy(self, data, *, byte_order=None):
        if isinstance(data, numpy.ndarray):
            if byte_order:
                dtype = data.dtype.newbyteorder(byte_order)
                data = numpy.asarray(data, dtype=dtype)
            return data
        else:
            raise ValueError('Cannot convert non-numpy from base Ops class')

    def minibatch(self, size: Union[int, Generator], sequence: Batchable, *, shuffle: bool=False, buffer: int=1) -> SizedGenerator:
        """Iterate slices from a sequence, optionally shuffled. Slices
        may be either views or copies of the underlying data.

        The `size` argument may be either an integer, or a sequence of integers.
        If a sequence, a new size is drawn before every output.

        If shuffle is True, shuffled batches are produced by first generating
        an index array, shuffling it, and then using it to slice into the
        sequence.

        An internal queue of `buffer` items is accumulated before being each
        output. Buffering is useful for some devices, to allow the
        network to run asynchronously without blocking on every batch.
        """
        if not hasattr(sequence, '__len__'):
            err = f"Can't minibatch data. Expected sequence, got {type(sequence)}"
            raise ValueError(err)
        sizes = self._get_batch_sizes(len(sequence), itertools.repeat(size) if isinstance(size, int) else size)
        indices = numpy.arange(len(sequence))

        def _iter_items():
            if shuffle:
                numpy.random.shuffle(indices)
            queue = []
            i = 0
            for size in sizes:
                size = int(size)
                queue.append(self._get_batch(sequence, indices[i:i + size]))
                if len(queue) >= buffer:
                    yield from queue
                    queue = []
                i += size
            yield from queue
        return SizedGenerator(_iter_items, len(sizes))

    def multibatch(self, size: Union[int, Generator], sequence: Batchable, *others: Batchable, shuffle: bool=False, buffer: int=1) -> SizedGenerator:
        """Minibatch one or more sequences of data, and yield
        lists with one batch per sequence. See ops.minibatch.
        """
        sequences = (sequence,) + tuple(others)
        if not all((hasattr(seq, '__len__') for seq in sequences)):
            values = ', '.join([f'{type(seq)}' for seq in sequences])
            err = f"Can't multibatch data. Expected sequences, got {values}"
            raise ValueError(err)
        sizes = self._get_batch_sizes(len(sequence), itertools.repeat(size) if isinstance(size, int) else size)
        indices = numpy.arange(len(sequence))

        def _iter_items():
            if shuffle:
                numpy.random.shuffle(indices)
            queue = []
            i = 0
            for size in sizes:
                size = int(size)
                idx_batch = indices[i:i + size]
                queue.append([])
                for sequence in sequences:
                    queue[-1].append(self._get_batch(sequence, idx_batch))
                if len(queue) >= buffer:
                    yield from queue
                    queue = []
                i += size
            yield from queue
        return SizedGenerator(_iter_items, len(sizes))

    def _get_batch(self, sequence, indices):
        if isinstance(sequence, list):
            subseq = [sequence[i] for i in indices]
        elif isinstance(sequence, tuple):
            subseq = tuple((sequence[i] for i in indices))
        else:
            subseq = sequence[indices]
        if is_xp_array(subseq):
            subseq = self.as_contig(self.xp.asarray(subseq))
        return subseq

    def _get_batch_sizes(self, length: int, sizes: Iterator[int]):
        output = []
        i = 0
        while i < length:
            output.append(next(sizes))
            i += output[-1]
        return output

    def seq2col(self, seq: Floats2d, nW: int, *, lengths: Optional[Ints1d]=None) -> Floats2d:
        """Given an (M, N) sequence of vectors, return an (M, N*(nW*2+1))
        sequence. The new sequence is constructed by concatenating nW preceding
        and succeeding vectors onto each column in the sequence, to extract a
        window of features.
        """
        assert nW == 1
        assert lengths == None
        B = seq.shape[0]
        I = seq.shape[1]
        cols = self.alloc3f(B, nW * 2 + 1, I)
        cols[nW:, :nW] = self.reshape3f(seq[:-nW], -1, nW, I)
        cols[:, nW] = seq
        cols[:-nW, nW + 1:] = self.reshape3f(seq[nW:], -1, nW, I)
        return self.reshape2f(cols, B, I * (2 * nW + 1))

    def backprop_seq2col(self, dY: Floats2d, nW: int, *, lengths: Optional[Ints1d]=None) -> Floats2d:
        """The reverse/backward operation of the `seq2col` function: calculate
        the gradient of the original `(M, N)` sequence, as a function of the
        gradient of the output `(M, N*(nW*2+1))` sequence.
        """
        assert nW == 1
        assert lengths == None
        nF = nW * 2 + 1
        B = dY.shape[0]
        I = dY.shape[1] // nF
        dX = self.alloc2f(B, I)
        dY3d = self.reshape3f(dY, B, nF, I)
        dX[:-nW] += self.reshape2f(dY3d[nW:, :nW], -1, I)
        dX += dY3d[:, nW]
        dX[nW:] += self.reshape2f(dY3d[:-nW, nW + 1:], -1, I)
        return dX

    def gemm(self, x: Floats2d, y: Floats2d, out: Optional[Floats2d]=None, trans1: bool=False, trans2: bool=False) -> Floats2d:
        """Perform General Matrix Multiplication (GeMM) and optionally store
        the result in the specified output variable.
        """
        if trans1:
            x = x.T
        if trans2:
            y = y.T
        if out is None:
            return self.xp.dot(x, y)
        else:
            self.xp.dot(x, y, out=out)
            return out

    def tile(self, X: Floats2d, reps: int) -> Floats2d:
        return self.xp.tile(X, reps)

    def affine(self, X: Floats2d, W: Floats2d, b: Floats1d) -> Floats2d:
        """Apply a weights layer and a bias to some inputs, i.e.
        Y = X @ W.T + b
        """
        Y = self.gemm(X, W, trans2=True)
        Y += b
        return Y

    @overload
    def flatten(self, X: List[Floats2d], dtype: Optional[DTypes]=None, pad: int=0, ndim_if_empty: int=2) -> Floats2d:
        ...

    @overload
    def flatten(self, X: List[Ints1d], dtype: Optional[DTypes]=None, pad: int=0, ndim_if_empty: int=2) -> Ints1d:
        ...

    @overload
    def flatten(self, X: List2d, dtype: Optional[DTypes]=None, pad: int=0, ndim_if_empty: int=2) -> Array2d:
        ...

    @overload
    def flatten(self, X: ListXd, dtype: Optional[DTypes]=None, pad: int=0, ndim_if_empty: int=2) -> ArrayXd:
        ...

    @overload
    def flatten(self, X: Sequence[ArrayXd], dtype: Optional[DTypes]=None, pad: int=0, ndim_if_empty: int=2) -> ArrayXd:
        ...

    def flatten(self, X: Sequence[ArrayXd], dtype: Optional[DTypes]=None, pad: int=0, ndim_if_empty: int=2) -> ArrayXd:
        """Flatten a list of arrays into one large array."""
        if X is None or len(X) == 0:
            return self.alloc((0,) * ndim_if_empty, dtype=dtype or 'f')
        xp = get_array_module(X[0])
        shape_if_empty = X[0].shape
        X = [x for x in X if x.size != 0]
        if len(X) == 0:
            return self.alloc(shape_if_empty, dtype=dtype or 'f')
        if int(pad) >= 1:
            padded = []
            for x in X:
                padded.append(xp.zeros((pad,) + x.shape[1:], dtype=x.dtype))
                padded.append(x)
            padded.append(xp.zeros((pad,) + x.shape[1:], dtype=x.dtype))
            X = padded
        result = xp.concatenate(X)
        if dtype is not None:
            result = xp.asarray(result, dtype=dtype)
        return result

    @overload
    def unflatten(self, X: Floats2d, lengths: Ints1d, pad: int=0) -> List[Floats2d]:
        ...

    @overload
    def unflatten(self, X: Ints1d, lengths: Ints1d, pad: int=0) -> List[Ints1d]:
        ...

    @overload
    def unflatten(self, X: Array2d, lengths: Ints1d, pad: int=0) -> List2d:
        ...

    @overload
    def unflatten(self, X: ArrayXd, lengths: Ints1d, pad: int=0) -> ListXd:
        ...

    def unflatten(self, X: ArrayXd, lengths: Ints1d, pad: int=0) -> ListXd:
        """The reverse/backward operation of the `flatten` function: unflatten
        a large array into a list of arrays according to the given lengths.
        """
        lengths = to_numpy(lengths)
        if pad > 0:
            lengths = numpy.where(lengths > 0, lengths + pad, 0)
        unflat = self.xp.split(X, numpy.cumsum(lengths))[:-1]
        if pad > 0:
            unflat = [a[pad:] for a in unflat]
        assert len(unflat) == len(lengths)
        return unflat

    @overload
    def pad(self, seqs: List[Ints2d], round_to=1) -> Ints3d:
        ...

    @overload
    def pad(self, seqs: List[Floats2d], round_to=1) -> Floats3d:
        ...

    def pad(self, seqs: Union[List[Ints2d], List[Floats2d]], round_to=1) -> Array3d:
        """Perform padding on a list of arrays so that they each have the same
        length, by taking the maximum dimension across each axis. This only
        works on non-empty sequences with the same `ndim` and `dtype`.
        """
        if round_to < 1:
            raise ValueError(f'Rounding for padding must at least be 1, was: {round_to}')
        if not seqs:
            raise ValueError('Cannot pad empty sequence')
        if len(set((seq.ndim for seq in seqs))) != 1:
            raise ValueError('Cannot pad sequences with different ndims')
        if len(set((seq.dtype for seq in seqs))) != 1:
            raise ValueError('Cannot pad sequences with different dtypes')
        if len(set((seq.shape[1:] for seq in seqs))) != 1:
            raise ValueError('Cannot pad sequences that differ on other dimensions')
        max_seq_len = max((len(seq) for seq in seqs))
        max_seq_len += -max_seq_len % round_to
        final_shape = (len(seqs), max_seq_len) + seqs[0].shape[1:]
        output: Array3d = cast(Array3d, self.alloc(final_shape, dtype=seqs[0].dtype))
        for i, arr in enumerate(seqs):
            output[i, :arr.shape[0]] = arr
        return output

    def unpad(self, padded: Array3d, lengths: List[int]) -> List2d:
        """The reverse/backward operation of the `pad` function: transform an
        array back into a list of arrays, each with their original length.
        """
        output = []
        for i, length in enumerate(lengths):
            output.append(padded[i, :length])
        return cast(List2d, output)

    def list2padded(self, seqs: List2d) -> Padded:
        """Pack a sequence of 2d arrays into a Padded datatype."""
        if not seqs:
            return Padded(self.alloc3f(0, 0, 0), self.alloc1i(0), self.alloc1i(0), self.alloc1i(0))
        elif len(seqs) == 1:
            data = self.reshape3(seqs[0], seqs[0].shape[0], 1, seqs[0].shape[1])
            size_at_t = self.asarray1i([1] * data.shape[0])
            lengths = self.asarray1i([data.shape[0]])
            indices = self.asarray1i([0])
            return Padded(data, size_at_t, lengths, indices)
        lengths_indices = [(len(seq), i) for i, seq in enumerate(seqs)]
        lengths_indices.sort(reverse=True)
        indices_ = [i for length, i in lengths_indices]
        lengths_ = [length for length, i in lengths_indices]
        nS = max([seq.shape[0] for seq in seqs])
        nB = len(seqs)
        nO = seqs[0].shape[1]
        seqs = cast(List2d, [seqs[i] for i in indices_])
        arr: Array3d = self.pad(seqs)
        assert arr.shape == (nB, nS, nO), (nB, nS, nO)
        arr = self.as_contig(arr.transpose((1, 0, 2)))
        assert arr.shape == (nS, nB, nO)
        batch_size_at_t_ = [0 for _ in range(nS)]
        current_size = len(lengths_)
        for t in range(nS):
            while current_size and t >= lengths_[current_size - 1]:
                current_size -= 1
            batch_size_at_t_[t] = current_size
        assert sum(lengths_) == sum(batch_size_at_t_)
        return Padded(arr, self.asarray1i(batch_size_at_t_), self.asarray1i(lengths_), self.asarray1i(indices_))

    def padded2list(self, padded: Padded) -> List2d:
        """Unpack a Padded datatype to a list of 2-dimensional arrays."""
        data = padded.data
        indices = to_numpy(padded.indices)
        lengths = to_numpy(padded.lengths)
        unpadded: List[Optional[Array2d]] = [None] * len(lengths)
        data = self.as_contig(data.transpose((1, 0, 2)))
        for i in range(data.shape[0]):
            unpadded[indices[i]] = data[i, :int(lengths[i])]
        return cast(List2d, unpadded)

    def get_dropout_mask(self, shape: Shape, drop: Optional[float]) -> FloatsXd:
        """Create a random mask for applying dropout, with a certain percent of
        the mask (defined by `drop`) will contain zeros. The neurons at those
        positions will be deactivated during training, resulting in a more
        robust network and less overfitting.
        """
        if drop is None or drop <= 0:
            return self.xp.ones(shape, dtype='f')
        elif drop >= 1.0:
            return self.alloc_f(shape)
        coinflips = self.xp.random.uniform(0.0, 1.0, shape)
        mask = (coinflips >= drop) / (1.0 - drop)
        return cast(FloatsXd, self.asarray(mask, dtype='float32'))

    def alloc1f(self, d0: int, *, dtype: Optional[DTypesFloat]='float32', zeros: bool=True) -> Floats1d:
        return cast(Floats1d, self.alloc((d0,), dtype=dtype, zeros=zeros))

    def alloc2f(self, d0: int, d1: int, *, dtype: Optional[DTypesFloat]='float32', zeros: bool=True) -> Floats2d:
        return cast(Floats2d, self.alloc((d0, d1), dtype=dtype, zeros=zeros))

    def alloc3f(self, d0: int, d1: int, d2: int, *, dtype: Optional[DTypesFloat]='float32', zeros: bool=True) -> Floats3d:
        return cast(Floats3d, self.alloc((d0, d1, d2), dtype=dtype, zeros=zeros))

    def alloc4f(self, d0: int, d1: int, d2: int, d3: int, *, dtype: Optional[DTypesFloat]='float32', zeros: bool=True) -> Floats4d:
        return cast(Floats4d, self.alloc((d0, d1, d2, d3), dtype=dtype, zeros=zeros))

    def alloc_f(self, shape: Shape, *, dtype: Optional[DTypesFloat]='float32', zeros: bool=True) -> FloatsXd:
        return cast(FloatsXd, self.alloc(shape, dtype=dtype, zeros=zeros))

    def alloc1i(self, d0: int, *, dtype: Optional[DTypesInt]='int32', zeros: bool=True) -> Ints1d:
        return cast(Ints1d, self.alloc((d0,), dtype=dtype, zeros=zeros))

    def alloc2i(self, d0: int, d1: int, *, dtype: Optional[DTypesInt]='int32', zeros: bool=True) -> Ints2d:
        return cast(Ints2d, self.alloc((d0, d1), dtype=dtype, zeros=zeros))

    def alloc3i(self, d0: int, d1: int, d2: int, *, dtype: Optional[DTypesInt]='int32', zeros: bool=True) -> Ints3d:
        return cast(Ints3d, self.alloc((d0, d1, d2), dtype=dtype, zeros=zeros))

    def alloc4i(self, d0: int, d1: int, d2: int, d3: int, *, dtype: Optional[DTypesInt]='int32', zeros: bool=True) -> Ints4d:
        return cast(Ints4d, self.alloc((d0, d1, d2, d3), dtype=dtype, zeros=zeros))

    def alloc_i(self, shape: Shape, *, dtype: Optional[DTypesInt]='int32', zeros: bool=True) -> IntsXd:
        return cast(IntsXd, self.alloc(shape, dtype=dtype, zeros=zeros))

    def alloc(self, shape: Shape, *, dtype: Optional[DTypes]='float32', zeros: bool=True) -> Any:
        """Allocate an array of a certain shape."""
        if isinstance(shape, int):
            shape = (shape,)
        if zeros:
            return self.xp.zeros(shape, dtype=dtype)
        else:
            return self.xp.empty(shape, dtype=dtype)

    def reshape1(self, array: ArrayXd, d0: int) -> Array1d:
        return cast(Array1d, self.reshape(array, (d0,)))

    def reshape2(self, array: ArrayXd, d0: int, d1: int) -> Array2d:
        return cast(Array2d, self.reshape(array, (d0, d1)))

    def reshape3(self, array: ArrayXd, d0: int, d1: int, d2: int) -> Array3d:
        return cast(Array3d, self.reshape(array, (d0, d1, d2)))

    def reshape4(self, array: ArrayXd, d0: int, d1: int, d2: int, d3: int) -> Array4d:
        return cast(Array4d, self.reshape(array, (d0, d1, d2, d3)))

    def reshape1f(self, array: FloatsXd, d0: int) -> Floats1d:
        return cast(Floats1d, self.reshape(array, (d0,)))

    def reshape2f(self, array: FloatsXd, d0: int, d1: int) -> Floats2d:
        return cast(Floats2d, self.reshape(array, (d0, d1)))

    def reshape3f(self, array: FloatsXd, d0: int, d1: int, d2: int) -> Floats3d:
        return cast(Floats3d, self.reshape(array, (d0, d1, d2)))

    def reshape4f(self, array: FloatsXd, d0: int, d1: int, d2: int, d3: int) -> Floats4d:
        return cast(Floats4d, self.reshape(array, (d0, d1, d2, d3)))

    def reshape_f(self, array: FloatsXd, shape: Shape) -> FloatsXd:
        return self.reshape(array, shape)

    def reshape1i(self, array: IntsXd, d0: int) -> Ints1d:
        return cast(Ints1d, self.reshape(array, (d0,)))

    def reshape2i(self, array: IntsXd, d0: int, d1: int) -> Ints2d:
        return cast(Ints2d, self.reshape(array, (d0, d1)))

    def reshape3i(self, array: IntsXd, d0: int, d1: int, d2: int) -> Ints3d:
        return cast(Ints3d, self.reshape(array, (d0, d1, d2)))

    def reshape4i(self, array: IntsXd, d0: int, d1: int, d2: int, d3: int) -> Ints4d:
        return cast(Ints4d, self.reshape(array, (d0, d1, d2, d3)))

    def reshape_i(self, array: IntsXd, shape: Shape) -> IntsXd:
        return self.reshape(array, shape)

    def reshape(self, array: ArrayT, shape: Shape) -> ArrayT:
        """Reshape an array."""
        if isinstance(shape, int):
            shape = (shape,)
        return cast(ArrayT, array.reshape(shape))

    def asarray4f(self, data: Union[Floats4d, Sequence[Sequence[Sequence[Sequence[float]]]]], *, dtype: Optional[DTypes]='float32') -> Floats4d:
        return cast(Floats4d, self.asarray(data, dtype=dtype))

    def asarray3f(self, data: Union[Floats3d, Sequence[Sequence[Sequence[float]]]], *, dtype: Optional[DTypes]='float32') -> Floats3d:
        return cast(Floats3d, self.asarray(data, dtype=dtype))

    def asarray2f(self, data: Union[Floats2d, Sequence[Sequence[float]]], *, dtype: Optional[DTypes]='float32') -> Floats2d:
        return cast(Floats2d, self.asarray(data, dtype=dtype))

    def asarray1f(self, data: Union[Floats1d, Sequence[float]], *, dtype: Optional[DTypes]='float32') -> Floats1d:
        return cast(Floats1d, self.asarray(data, dtype=dtype))

    def asarray_f(self, data: Union[FloatsXd, Sequence[Any]], *, dtype: Optional[DTypes]='float32') -> FloatsXd:
        return cast(FloatsXd, self.asarray(data, dtype=dtype))

    def asarray1i(self, data: Union[Ints1d, Sequence[int]], *, dtype: Optional[DTypes]='int32') -> Ints1d:
        return cast(Ints1d, self.asarray(data, dtype=dtype))

    def asarray2i(self, data: Union[Ints2d, Sequence[Sequence[int]]], *, dtype: Optional[DTypes]='int32') -> Ints2d:
        return cast(Ints2d, self.asarray(data, dtype=dtype))

    def asarray3i(self, data: Union[Ints3d, Sequence[Sequence[Sequence[int]]]], *, dtype: Optional[DTypes]='int32') -> Ints3d:
        return cast(Ints3d, self.asarray(data, dtype=dtype))

    def asarray4i(self, data: Union[Ints4d, Sequence[Sequence[Sequence[Sequence[int]]]]], *, dtype: Optional[DTypes]='int32') -> Ints4d:
        return cast(Ints4d, self.asarray(data, dtype=dtype))

    def asarray_i(self, data: Union[IntsXd, Sequence[Any]], *, dtype: Optional[DTypes]='int32') -> IntsXd:
        return cast(IntsXd, self.asarray(data, dtype=dtype))

    def asarray(self, data: Union[ArrayXd, Sequence[ArrayXd], Sequence[Any]], *, dtype: Optional[DTypes]=None) -> ArrayXd:
        """Ensure a given array is of the correct type."""
        if isinstance(data, self.xp.ndarray):
            if dtype is None:
                return data
            elif data.dtype == dtype:
                return data
            else:
                return self.xp.asarray(data, dtype=dtype)
        elif hasattr(data, 'numpy'):
            return data.numpy()
        elif dtype is not None:
            return self.xp.array(data, dtype=dtype)
        else:
            return self.xp.array(data)

    def as_contig(self, data: ArrayT, dtype: Optional[DTypes]=None) -> ArrayT:
        """Allow the backend to make a contiguous copy of an array.
        Implementations of `Ops` do not have to make a copy or make it
        contiguous if that would not improve efficiency for the execution engine.
        """
        if data.flags['C_CONTIGUOUS'] and dtype in (None, data.dtype):
            return data
        kwargs = {'dtype': dtype} if dtype is not None else {}
        return self.xp.ascontiguousarray(data, **kwargs)

    def sigmoid(self, X: FloatsXdT, *, inplace: bool=False) -> FloatsXdT:
        if inplace:
            X = self.xp.clip(X, -20.0, 20.0, out=X)
            self.xp.exp(-X, out=X)
            X += 1.0
            X **= -1.0
            return X
        else:
            X = self.xp.clip(X, -20.0, 20.0)
            return 1.0 / (1.0 + self.xp.exp(-X))

    def backprop_sigmoid(self, dY: FloatsXdT, Y: FloatsXdT, *, inplace: bool=False) -> FloatsXdT:
        if inplace:
            self.dsigmoid(Y, inplace=True)
            Y *= dY
            return Y
        else:
            return dY * self.dsigmoid(Y, inplace=inplace)

    def dsigmoid(self, Y: FloatsXdT, *, inplace: bool=False) -> FloatsXdT:
        if inplace:
            Y *= 1 - Y
            return Y
        else:
            return Y * (1.0 - Y)

    def dtanh(self, Y: FloatsT, *, inplace: bool=False) -> FloatsT:
        if inplace:
            Y **= 2
            Y *= -1.0
            Y += 1.0
            return Y
        else:
            return 1 - Y ** 2

    def softmax(self, x: FloatsT, *, inplace: bool=False, axis: int=-1, temperature: float=1.0) -> FloatsT:
        if temperature != 1.0:
            x = x / temperature
        maxes = self.xp.max(x, axis=axis, keepdims=True)
        shifted = x - maxes
        new_x = self.xp.exp(shifted)
        new_x /= new_x.sum(axis=axis, keepdims=True)
        return new_x

    def softmax_sequences(self, Xs: Floats2d, lengths: Ints1d, *, inplace: bool=False, axis: int=-1) -> Floats2d:
        if Xs.ndim >= 3:
            err = f'Softmax currently only supports 2d. Got: {Xs.ndim}'
            raise NotImplementedError(err)
        Xs = self.xp.clip(Xs, -20.0, 20.0)
        new_x = self.xp.exp(Xs)
        summed = self.backprop_reduce_sum(self.reduce_sum(new_x, lengths), lengths)
        new_x /= summed
        return new_x

    def backprop_softmax(self, Y: FloatsT, dY: FloatsT, *, axis: int=-1, temperature: float=1.0) -> FloatsT:
        if temperature != 1.0:
            dY = dY / temperature
        dX = Y * dY
        dX -= Y * dX.sum(axis=axis, keepdims=True)
        return dX

    def backprop_softmax_sequences(self, dY: Floats2d, Y: Floats2d, lengths: Ints1d) -> Floats2d:
        dX = Y * dY
        sum_dX = self.backprop_reduce_sum(self.reduce_sum(dX, lengths), lengths)
        dX -= Y * sum_dX
        return dX

    def lstm_forward_training(self, params: Floats1d, H0: Floats3d, C0: Floats3d, X: Floats2d, size_at_t: Ints1d) -> Tuple[Floats2d, Tuple]:
        assert H0.shape == C0.shape
        assert H0.shape[1] == C0.shape[1]
        Y, fwd_state = lstm_forward_training(params, H0, C0, X, size_at_t)
        return (Y, fwd_state)

    def lstm_forward_inference(self, params: Floats1d, H0: Floats3d, C0: Floats3d, X: Floats2d, size_at_t: Ints1d) -> Floats2d:
        Y, _ = lstm_forward_training(params, H0, C0, X, size_at_t)
        return Y

    def backprop_lstm(self, dY: Floats2d, lengths: Ints1d, params: Floats1d, fwd_state: Tuple) -> Tuple[Floats2d, Floats1d]:
        dX, d_params = backprop_lstm(dY, lengths, params, fwd_state)
        return (dX, d_params)

    def maxout(self, X: Floats3d) -> Tuple[Floats2d, Ints2d]:
        which = X.argmax(axis=-1)
        return (X.max(axis=-1), which)

    def backprop_maxout(self, dY: Floats2d, which: Ints2d, P: int) -> Floats3d:
        dX = self.alloc3f(dY.shape[0], dY.shape[1], P, dtype=dY.dtype)
        for b in range(dY.shape[0]):
            for o in range(dY.shape[1]):
                dX[b, o, which[b, o]] = dY[b, o]
        return dX

    def relu(self, X: Floats2d, inplace: bool=False) -> Floats2d:
        if not inplace:
            return X * (X > 0)
        else:
            X *= X > 0
            return X

    def backprop_relu(self, dY: Floats2d, Y: Floats2d, inplace: bool=False) -> Floats2d:
        if not inplace:
            return dY * (Y > 0)
        dY *= Y > 0
        return dY

    def clipped_linear(self, X: FloatsXdT, slope: float=1.0, offset: float=0.0, min_val: float=0.0, max_val: float=1.0, inplace: bool=False) -> FloatsXdT:
        if inplace:
            X *= slope
            X += offset
            return self.xp.clip(X, min_val, max_val, out=X)
        out = X * slope + offset
        return self.xp.clip(out, min_val, max_val)

    def backprop_clipped_linear(self, dY: FloatsXdT, X: FloatsXdT, slope: float=1.0, offset: float=0.0, min_val: float=0.0, max_val: float=1.0, inplace: bool=False) -> FloatsXdT:
        low = (min_val - offset) / slope
        high = (max_val - offset) / slope
        slope = self.xp.float64(slope).astype(X.dtype)
        zero = self.xp.float64(0.0).astype(X.dtype)
        dX = self.xp.where((low < X) & (X < high), slope, zero)
        if inplace:
            dY *= dX
            return dY
        return dY * dX

    def relu_k(self, X: FloatsXdT, n: float=6.0, inplace: bool=False) -> FloatsXdT:
        return self.clipped_linear(X, max_val=n, inplace=inplace)

    def backprop_relu_k(self, dY: FloatsXdT, X: FloatsXdT, n: float=6.0, inplace: bool=False) -> FloatsXdT:
        return self.backprop_clipped_linear(dY, X, max_val=n, inplace=inplace)

    def hard_sigmoid(self, X: FloatsXdT, inplace: bool=False) -> FloatsXdT:
        return self.clipped_linear(X, slope=0.2, offset=0.5, inplace=inplace)

    def backprop_hard_sigmoid(self, dY: FloatsXdT, X: FloatsXdT, inplace: bool=False) -> FloatsXdT:
        return self.backprop_clipped_linear(dY, X, slope=0.2, offset=0.5)

    def hard_tanh(self, X: FloatsXdT, inplace: bool=False) -> FloatsXdT:
        return self.clipped_linear(X, min_val=-1.0, max_val=1.0, inplace=inplace)

    def backprop_hard_tanh(self, dY: FloatsXdT, X: FloatsXdT, inplace: bool=False) -> FloatsXdT:
        return self.backprop_clipped_linear(dY, X, min_val=-1.0, max_val=1.0)

    def swish(self, X: FloatsXdT, inplace: bool=False) -> FloatsXdT:
        if inplace:
            X *= self.sigmoid(X)
            return X
        out = X * self.sigmoid(X)
        return out

    def backprop_swish(self, dY: FloatsXdT, X: FloatsXdT, Y: FloatsXdT, inplace: bool=False) -> FloatsXdT:
        Y = Y + self.sigmoid(X) * (1 - Y)
        if inplace:
            dY *= Y
            return dY
        out = dY * Y
        return out

    def hard_swish(self, X: FloatsXdT, inplace: bool=False) -> FloatsXdT:
        if inplace:
            X *= self.hard_sigmoid(X)
            return X
        out = X * self.hard_sigmoid(X)
        return out

    def backprop_hard_swish(self, dY: FloatsXdT, X: FloatsXdT, inplace: bool=False) -> FloatsXdT:
        dX = X * 0.4 + 0.5
        dX[X > 2.5] = 1.0
        dX[X < -2.5] = 0
        if inplace:
            dY *= dX
            return dY
        return dY * dX

    def hard_swish_mobilenet(self, X: FloatsXdT, inplace: bool=False) -> FloatsXdT:
        if inplace:
            X *= self.relu_k(X + 3) / 6
            return X
        return X * (self.relu_k(X + 3) / 6)

    def backprop_hard_swish_mobilenet(self, dY: FloatsXdT, X: FloatsXdT, inplace: bool=False) -> FloatsXdT:
        dX = 1 / 6 * (X * 2.0 + 3.0)
        dX[X > 3.0] = 1.0
        dX[X < -3.0] = 0
        if inplace:
            dY *= dX
            return dY
        return dX * dY

    def dish(self, X: FloatsXdT, inplace: bool=False) -> FloatsXdT:
        tmp = self.xp.square(X)
        tmp += 1.0
        self.xp.sqrt(tmp, out=tmp)
        tmp = X / tmp
        tmp += 1
        tmp *= 0.5
        if inplace:
            X *= tmp
            return X
        else:
            return X * tmp

    def backprop_dish(self, dY: FloatsXdT, X: FloatsXdT, inplace: bool=False) -> FloatsXdT:
        x_sq = self.xp.square(X)
        x_sq_plus_one = x_sq + 1.0
        deriv = X / self.xp.sqrt(x_sq_plus_one)
        second = 0.5 * X * x_sq
        second /= x_sq_plus_one ** 1.5
        deriv -= second
        deriv += 0.5
        if inplace:
            dY *= deriv
            return dY
        else:
            return dY * deriv

    def erf(self, X: FloatsXdT) -> FloatsXdT:
        sign = self.xp.sign(X)
        X = self.xp.abs(X)
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911
        t = 1.0 / (1.0 + p * X)
        y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * self.xp.exp(-X * X)
        out = sign * y
        out = out.astype(X.dtype)
        return out

    def sechsq(self, X: FloatsXdT) -> FloatsXdT:
        X = self.xp.clip(X, -20.0, 20.0)
        return (1 / self.xp.cosh(X)) ** 2

    def gelu_approx(self, X: FloatsXdT, inplace: bool=False) -> FloatsXdT:
        tmp = 1.0 + self.xp.tanh(SQRT2PI * (X + 0.044715 * self.xp.power(X, 3)))
        tmp *= 0.5
        tmp = tmp.astype(X.dtype)
        if inplace:
            X *= tmp
            return X
        else:
            Y = self.xp.array(X)
            Y *= tmp
            return Y

    def backprop_gelu_approx(self, dY: FloatsXdT, X: FloatsXdT, inplace: bool=False) -> FloatsXdT:
        dX = cast(FloatsXdT, self.alloc_f(X.shape))
        Xp3 = self.xp.power(X, 3)
        tmp = 0.5 * self.xp.tanh(0.0356774 * Xp3 + 0.797885 * X)
        tmp += (0.0535161 * Xp3 + 0.398942 * X) * self.sechsq(0.0356774 * Xp3 + 0.797885 * X)
        tmp += 0.5
        dX += tmp
        if inplace:
            dY *= dX
            return dY
        return dY * dX

    def gelu(self, X: FloatsXdT, inplace: bool=False) -> FloatsXdT:
        cdf = gaussian_cdf(self, X)
        if inplace:
            X *= cdf
            return X
        return X * cdf

    def backprop_gelu(self, dY: FloatsXdT, X: FloatsXdT, inplace: bool=False) -> FloatsXdT:
        dX = gaussian_cdf(self, X) + X * gaussian_pdf(self, X)
        if inplace:
            dY *= dX
            return dY
        return dY * dX

    def mish(self, X: FloatsXdT, threshold: float=20.0, inplace: bool=False) -> FloatsXdT:
        tmp = X * self.xp.tanh(self.xp.log(1.0 + self.xp.exp(X)))
        Y = self.xp.where(X >= threshold, X, tmp)
        if inplace:
            X[:] = Y
            return X
        else:
            return Y

    def backprop_mish(self, dY: FloatsXdT, X: Floats2d, threshold: float=20.0, inplace: bool=False) -> FloatsXdT:
        if dY.shape != X.shape:
            msg = f'arrays have incompatible shapes: {dY.shape} and {X.shape}'
            raise ValueError(msg)
        xp = get_array_module(X)
        indices = X < threshold
        Xsub = X[indices]
        dYsub = dY[indices]
        omega = 4.0 * (Xsub + 1.0)
        omega += 4.0 * xp.exp(2.0 * Xsub)
        omega += xp.exp(3.0 * Xsub)
        omega += xp.exp(Xsub) * (4.0 * Xsub + 6.0)
        delta = xp.exp(Xsub) + 1.0
        delta *= delta
        delta += 1.0
        dXsub = dYsub * (xp.exp(Xsub) * omega / delta ** 2)
        if inplace:
            out = dY
        else:
            out = xp.copy(dY)
        out[indices] = dXsub
        return out

    def update_averages(self, ema: FloatsT, weights: FloatsT, t: int, max_decay: float=0.9999) -> None:
        decay = (1.0 + t) / (10.0 + t)
        if decay > max_decay:
            decay = max_decay
        ema -= (1 - decay) * (ema - weights)

    def adam(self, weights: Floats1d, gradient: Floats1d, mom1: Floats1d, mom2: Floats1d, beta1: float, beta2: float, eps: float, learn_rate: float, mod_rate: float=1.0) -> Tuple[Floats1d, Floats1d, Floats1d, Floats1d]:
        _check_compatible_shape(weights, gradient)
        _check_compatible_shape(weights, mom1)
        _check_compatible_shape(weights, mom2)
        mom1 *= beta1
        mom2 *= beta2
        mom1 += gradient * (1.0 - beta1)
        mom2 += gradient * gradient * (1.0 - beta2)
        weights -= learn_rate * (mom1 / (mod_rate * self.xp.sqrt(mom2) + eps))
        return (weights, gradient, mom1, mom2)

    def clip_gradient(self, gradient: FloatsT, threshold: float) -> FloatsT:
        xp = get_array_module(gradient)
        grad_norm = xp.linalg.norm(gradient)
        if grad_norm >= threshold:
            gradient *= threshold / grad_norm
        return gradient

    def logloss(self, y_true: FloatsT, y_pred: FloatsT) -> float:
        log_yp = self.xp.log(y_pred + 1e-08)
        loss = y_true * log_yp + (1 - y_true) * self.xp.log(1 - y_pred + 1e-08)
        return -loss

    def reduce_sum(self, X: Floats2d, lengths: Ints1d) -> Floats2d:
        Y = self.alloc2f(lengths.shape[0], X.shape[1], zeros=False)
        start = 0
        for i, length in enumerate(lengths):
            if length < 0:
                raise ValueError(f'all sequence lengths must be >= 0, got {length}')
            elif start + length > X.shape[0]:
                raise IndexError('lengths must sum up to the number of rows')
            elif length:
                Y[i] = X[start:start + length].sum(axis=0)
                start += length
            else:
                Y[i] = 0.0
        return Y

    def reduce_first(self, X: Floats2d, lengths: Ints1d) -> Tuple[Floats2d, Ints1d]:
        if lengths.size == 0:
            return (self.alloc2f(0, X.shape[1]), lengths)
        if not self.xp.all(lengths > 0):
            raise ValueError(f'all sequence lengths must be > 0')
        starts_ends = self.alloc1i(lengths.shape[0] + 1, zeros=False)
        starts_ends[0] = 0
        starts_ends[1:] = lengths.cumsum()
        if starts_ends[-1] != X.shape[0]:
            raise IndexError('lengths must sum up to the number of rows')
        return (X[starts_ends[:-1]], starts_ends)

    def reduce_last(self, X: Floats2d, lengths: Ints1d) -> Tuple[Floats2d, Ints1d]:
        if lengths.size == 0:
            return (self.alloc2f(0, X.shape[1]), lengths)
        if not self.xp.all(lengths > 0):
            raise ValueError(f'all sequence lengths must be > 0')
        lasts = lengths.cumsum() - 1
        if lasts[-1] + 1 != X.shape[0]:
            raise IndexError('lengths must sum up to the number of rows')
        return (X[lasts], lasts)

    def reduce_mean(self, X: Floats2d, lengths: Ints1d) -> Floats2d:
        Y = self.alloc2f(lengths.shape[0], X.shape[1], zeros=False)
        start = 0
        for i, length in enumerate(lengths):
            if length < 0:
                raise ValueError(f'all sequence lengths must be >= 0, got {length}')
            elif start + length > X.shape[0]:
                raise IndexError('lengths must sum up to the number of rows')
            elif length:
                Y[i] = X[start:start + length].mean(axis=0)
            else:
                Y[i] = 0.0
            start += length
        return Y

    def reduce_max(self, X: Floats2d, lengths: Ints1d) -> Tuple[Floats2d, Ints2d]:
        Y = self.alloc2f(lengths.shape[0], X.shape[1], dtype=X.dtype, zeros=False)
        which = self.alloc2i(lengths.shape[0], X.shape[1], zeros=False)
        start = 0
        for i, length in enumerate(lengths):
            if length <= 0:
                raise ValueError(f'all sequence lengths must be > 0, got {length}')
            elif start + length > X.shape[0]:
                raise IndexError('lengths must sum up to the number of rows')
            elif length:
                which[i] = X[start:start + length].argmax(axis=0)
                Y[i] = X[start:start + length].max(axis=0)
            start += length
        return (Y, which)

    def backprop_reduce_first(self, d_firsts: Floats2d, starts_ends: Ints1d) -> Floats2d:
        if starts_ends.size == 0:
            return self.alloc2f(0, d_firsts.shape[1], dtype=d_firsts.dtype, zeros=True)
        elif starts_ends.size == 1:
            raise ValueError(f'starts_ends must not have size 1')
        dX = self.alloc2f(int(starts_ends[-1]), d_firsts.shape[1], dtype=d_firsts.dtype, zeros=True)
        dX[starts_ends[:-1]] = d_firsts
        return dX

    def backprop_reduce_last(self, d_lasts: Floats2d, lasts: Ints1d) -> Floats2d:
        if lasts.size == 0:
            return self.alloc2f(0, d_lasts.shape[1], dtype=d_lasts.dtype, zeros=True)
        dX = self.alloc2f(int(lasts[-1]) + 1, d_lasts.shape[1], dtype=d_lasts.dtype, zeros=True)
        dX[lasts] = d_lasts
        return dX

    def backprop_reduce_sum(self, d_sums: Floats2d, lengths: Ints1d) -> Floats2d:
        dX = self.alloc2f(lengths.sum(), d_sums.shape[1], dtype=d_sums.dtype, zeros=False)
        start = 0
        for i, length in enumerate(lengths):
            if length < 0:
                raise ValueError(f'all sequence lengths must be >= 0, got {length}')
            dX[start:start + length] = d_sums[i]
            start += length
        return dX

    def backprop_reduce_mean(self, d_means: Floats2d, lengths: Ints1d) -> Floats2d:
        dX = self.alloc2f(lengths.sum(), d_means.shape[1], dtype=d_means.dtype, zeros=False)
        start = 0
        for i, length in enumerate(lengths):
            if length < 0:
                raise ValueError(f'all sequence lengths must be >= 0, got {length}')
            dX[start:start + length] = d_means[i] / length
            start += length
        return dX

    def backprop_reduce_max(self, d_maxes: Floats2d, which: Ints2d, lengths: Ints1d) -> Floats2d:
        dX = self.alloc2f(lengths.sum(), d_maxes.shape[1], dtype=d_maxes.dtype)
        start = 0
        for i, length in enumerate(lengths):
            if length <= 0:
                raise ValueError(f'all sequence lengths must be > 0, got {length}')
            self.xp.put_along_axis(dX[start:start + length], which[i].reshape((1, -1)), d_maxes[i], 0)
            start += length
        return dX

    def hash(self, ids: Ints1d, seed: int) -> Ints2d:
        """Hash a sequence of 64-bit keys into a table with 4 32-bit keys, using
        murmurhash3.
        """
        from .numpy_ops import NumpyOps
        numpy_ops = NumpyOps()
        return self.asarray2i(numpy_ops.hash(numpy_ops.asarray(ids, dtype='uint64'), seed))

    def ngrams(self, n: int, keys: Ints1d) -> Ints1d:
        from .numpy_ops import NumpyOps
        numpy_ops = NumpyOps()
        return self.asarray1i(numpy_ops.ngrams(n, numpy_ops.asarray(keys, dtype='uint64')))

    def position_encode(self, N: int, D: int, period: int=10000, out: Optional[Floats2d]=None) -> Floats2d:
        from .numpy_ops import NumpyOps
        numpy_ops = NumpyOps()
        return self.asarray2f(numpy_ops.position_encode(N, D, period, out))

    def gather_add(self, table: Floats2d, indices: Ints2d) -> Floats2d:
        return table[indices].sum(axis=1)

    def scatter_add(self, table: FloatsXd, indices: IntsXd, values: FloatsXd) -> FloatsXd:
        return self.xp.add.at(table, indices, values)

    def insert_into(self, shape, Xs):
        """Maybe don't need this? Just a quicky to get Jax working."""
        output = self.alloc(shape, dtype=Xs[0].dtype)
        for i, x in enumerate(Xs):
            output[i, :x.shape[0]] = x
        return output