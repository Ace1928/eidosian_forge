import unittest
from collections import namedtuple as nt
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_equal
from ..optpkg import optional_package
from ..viewers import OrthoSlicer3D
@needs_mpl
def test_viewer():
    plt = optional_package('matplotlib.pyplot')[0]
    a = np.sin(np.linspace(0, np.pi, 20))
    b = np.sin(np.linspace(0, np.pi * 5, 30))
    data = (np.outer(a, b)[..., np.newaxis] * a)[:, :, :, np.newaxis]
    data = data * np.array([1.0, 2.0])
    v = OrthoSlicer3D(data)
    assert_array_equal(v.position, (0, 0, 0))
    assert 'OrthoSlicer3D' in repr(v)
    v._on_scroll(nt('event', 'button inaxes key')('up', None, None))
    for ax in (v._axes[0], v._axes[3]):
        v._on_scroll(nt('event', 'button inaxes key')('up', ax, None))
    v._on_scroll(nt('event', 'button inaxes key')('up', ax, 'shift'))
    v._on_mouse(nt('event', 'xdata ydata inaxes button')(0.5, 0.5, None, 1))
    for ax in v._axes:
        v._on_mouse(nt('event', 'xdata ydata inaxes button')(0.5, 0.5, ax, 1))
    v._on_mouse(nt('event', 'xdata ydata inaxes button')(0.5, 0.5, None, None))
    v.set_volume_idx(1)
    v.cmap = 'hot'
    v.clim = (0, 3)
    with pytest.raises(ValueError):
        OrthoSlicer3D.clim.fset(v, (0.0,))
    with pytest.raises((ValueError, KeyError)):
        OrthoSlicer3D.cmap.fset(v, 'foo')
    v.set_volume_idx(1)
    v._on_keypress(nt('event', 'key')('-'))
    assert_equal(v._data_idx[3], 0)
    v._on_keypress(nt('event', 'key')('+'))
    assert_equal(v._data_idx[3], 1)
    v._on_keypress(nt('event', 'key')('-'))
    v._on_keypress(nt('event', 'key')('='))
    assert_equal(v._data_idx[3], 1)
    v.close()
    v._draw()
    v = OrthoSlicer3D(data[:, :, :, 0])
    v._on_scroll(nt('event', 'button inaxes key')('up', v._axes[0], 'shift'))
    v._on_keypress(nt('event', 'key')('escape'))
    v.close()
    with pytest.raises(TypeError):
        OrthoSlicer3D(data[:, :, :, 0].astype(np.complex64))
    fig, axes = plt.subplots(1, 4)
    plt.close(fig)
    v1 = OrthoSlicer3D(data, axes=axes)
    aff = np.array([[0, 1, 0, 3], [-1, 0, 0, 2], [0, 0, 2, 1], [0, 0, 0, 1]], float)
    v2 = OrthoSlicer3D(data, affine=aff, axes=axes[:3])
    with pytest.raises(ValueError):
        OrthoSlicer3D(data[:, :, 0, 0])
    with pytest.raises(ValueError):
        OrthoSlicer3D(data, affine=np.eye(3))
    with pytest.raises(TypeError):
        v2.link_to(1)
    v2.link_to(v1)
    v2.link_to(v1)
    v1.close()
    v2.close()