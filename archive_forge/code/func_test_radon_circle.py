import itertools
import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import run_in_parallel
from skimage._shared.utils import _supported_float_type, convert_to_float
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, iradon_sart, rescale
def test_radon_circle():
    a = np.ones((10, 10))
    with expected_warnings(['reconstruction circle']):
        radon(a, circle=True)
    shape = (61, 79)
    c0, c1 = np.ogrid[0:shape[0], 0:shape[1]]
    r = np.sqrt((c0 - shape[0] // 2) ** 2 + (c1 - shape[1] // 2) ** 2)
    radius = min(shape) // 2
    image = np.clip(radius - r, 0, np.inf)
    image = _rescale_intensity(image)
    angles = np.linspace(0, 180, min(shape), endpoint=False)
    sinogram = radon(image, theta=angles, circle=True)
    assert np.all(sinogram.std(axis=1) < 0.01)
    image = _random_circle(shape)
    sinogram = radon(image, theta=angles, circle=True)
    mass = sinogram.sum(axis=0)
    average_mass = mass.mean()
    relative_error = np.abs(mass - average_mass) / average_mass
    print(relative_error.max(), relative_error.mean())
    assert np.all(relative_error < 0.0032)