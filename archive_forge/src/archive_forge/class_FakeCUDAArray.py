from contextlib import contextmanager
import numpy as np
from_record_like = None
class FakeCUDAArray(object):
    """
    Implements the interface of a DeviceArray/DeviceRecord, but mostly just
    wraps a NumPy array.
    """
    __cuda_ndarray__ = True

    def __init__(self, ary, stream=0):
        self._ary = ary
        self.stream = stream

    @property
    def alloc_size(self):
        return self._ary.nbytes

    @property
    def nbytes(self):
        return self._ary.nbytes

    def __getattr__(self, attrname):
        try:
            attr = getattr(self._ary, attrname)
            return attr
        except AttributeError as e:
            msg = "Wrapped array has no attribute '%s'" % attrname
            raise AttributeError(msg) from e

    def bind(self, stream=0):
        return FakeCUDAArray(self._ary, stream)

    @property
    def T(self):
        return self.transpose()

    def transpose(self, axes=None):
        return FakeCUDAArray(np.transpose(self._ary, axes=axes))

    def __getitem__(self, idx):
        ret = self._ary.__getitem__(idx)
        if type(ret) not in [np.ndarray, np.void]:
            return ret
        else:
            return FakeCUDAArray(ret, stream=self.stream)

    def __setitem__(self, idx, val):
        return self._ary.__setitem__(idx, val)

    def copy_to_host(self, ary=None, stream=0):
        if ary is None:
            ary = np.empty_like(self._ary)
        else:
            check_array_compatibility(self, ary)
        np.copyto(ary, self._ary)
        return ary

    def copy_to_device(self, ary, stream=0):
        """
        Copy from the provided array into this array.

        This may be less forgiving than the CUDA Python implementation, which
        will copy data up to the length of the smallest of the two arrays,
        whereas this expects the size of the arrays to be equal.
        """
        sentry_contiguous(self)
        self_core, ary_core = (array_core(self), array_core(ary))
        if isinstance(ary, FakeCUDAArray):
            sentry_contiguous(ary)
            check_array_compatibility(self_core, ary_core)
        else:
            ary_core = np.array(ary_core, order='C' if self_core.flags['C_CONTIGUOUS'] else 'F', subok=True, copy=False)
            check_array_compatibility(self_core, ary_core)
        np.copyto(self_core._ary, ary_core)

    @property
    def shape(self):
        return FakeShape(self._ary.shape)

    def ravel(self, *args, **kwargs):
        return FakeCUDAArray(self._ary.ravel(*args, **kwargs))

    def reshape(self, *args, **kwargs):
        return FakeCUDAArray(self._ary.reshape(*args, **kwargs))

    def view(self, *args, **kwargs):
        return FakeCUDAArray(self._ary.view(*args, **kwargs))

    def is_c_contiguous(self):
        return self._ary.flags.c_contiguous

    def is_f_contiguous(self):
        return self._ary.flags.f_contiguous

    def __str__(self):
        return str(self._ary)

    def __repr__(self):
        return repr(self._ary)

    def __len__(self):
        return len(self._ary)

    def __eq__(self, other):
        return FakeCUDAArray(self._ary == other)

    def __ne__(self, other):
        return FakeCUDAArray(self._ary != other)

    def __lt__(self, other):
        return FakeCUDAArray(self._ary < other)

    def __le__(self, other):
        return FakeCUDAArray(self._ary <= other)

    def __gt__(self, other):
        return FakeCUDAArray(self._ary > other)

    def __ge__(self, other):
        return FakeCUDAArray(self._ary >= other)

    def __add__(self, other):
        return FakeCUDAArray(self._ary + other)

    def __sub__(self, other):
        return FakeCUDAArray(self._ary - other)

    def __mul__(self, other):
        return FakeCUDAArray(self._ary * other)

    def __floordiv__(self, other):
        return FakeCUDAArray(self._ary // other)

    def __truediv__(self, other):
        return FakeCUDAArray(self._ary / other)

    def __mod__(self, other):
        return FakeCUDAArray(self._ary % other)

    def __pow__(self, other):
        return FakeCUDAArray(self._ary ** other)

    def split(self, section, stream=0):
        return [FakeCUDAArray(a) for a in np.split(self._ary, range(section, len(self), section))]