import inspect
from functools import wraps
from math import atan2
from math import pi as PI
from math import sqrt
from warnings import warn
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.distance import pdist
from . import _moments
from ._find_contours import find_contours
from ._marching_cubes_lewiner import marching_cubes
from ._regionprops_utils import (
class RegionProperties:
    """Please refer to `skimage.measure.regionprops` for more information
    on the available region properties.
    """

    def __init__(self, slice, label, label_image, intensity_image, cache_active, *, extra_properties=None, spacing=None, offset=None):
        if intensity_image is not None:
            ndim = label_image.ndim
            if not (intensity_image.shape[:ndim] == label_image.shape and intensity_image.ndim in [ndim, ndim + 1]):
                raise ValueError('Label and intensity image shapes must match, except for channel (last) axis.')
            multichannel = label_image.shape < intensity_image.shape
        else:
            multichannel = False
        self.label = label
        if offset is None:
            offset = np.zeros((label_image.ndim,), dtype=int)
        self._offset = np.array(offset)
        self._slice = slice
        self.slice = slice
        self._label_image = label_image
        self._intensity_image = intensity_image
        self._cache_active = cache_active
        self._cache = {}
        self._ndim = label_image.ndim
        self._multichannel = multichannel
        self._spatial_axes = tuple(range(self._ndim))
        if spacing is None:
            spacing = np.full(self._ndim, 1.0)
        self._spacing = _normalize_spacing(spacing, self._ndim)
        self._pixel_area = np.prod(self._spacing)
        self._extra_properties = {}
        if extra_properties is not None:
            for func in extra_properties:
                name = func.__name__
                if hasattr(self, name):
                    msg = f"Extra property '{name}' is shadowed by existing property and will be inaccessible. Consider renaming it."
                    warn(msg)
            self._extra_properties = {func.__name__: func for func in extra_properties}

    def __getattr__(self, attr):
        if self._intensity_image is None and attr in _require_intensity_image:
            raise AttributeError(f"Attribute '{attr}' unavailable when `intensity_image` has not been specified.")
        if attr in self._extra_properties:
            func = self._extra_properties[attr]
            n_args = _infer_number_of_required_args(func)
            if n_args == 2:
                if self._intensity_image is not None:
                    if self._multichannel:
                        multichannel_list = [func(self.image, self.image_intensity[..., i]) for i in range(self.image_intensity.shape[-1])]
                        return np.stack(multichannel_list, axis=-1)
                    else:
                        return func(self.image, self.image_intensity)
                else:
                    raise AttributeError(f'intensity image required to calculate {attr}')
            elif n_args == 1:
                return func(self.image)
            else:
                raise AttributeError(f"Custom regionprop function's number of arguments must be 1 or 2, but {attr} takes {n_args} arguments.")
        elif attr in PROPS and attr.lower() == attr:
            if self._intensity_image is None and PROPS[attr] in _require_intensity_image:
                raise AttributeError(f"Attribute '{attr}' unavailable when `intensity_image` has not been specified.")
            return getattr(self, PROPS[attr])
        else:
            raise AttributeError(f"'{type(self)}' object has no attribute '{attr}'")

    def __setattr__(self, name, value):
        if name in PROPS:
            super().__setattr__(PROPS[name], value)
        else:
            super().__setattr__(name, value)

    @property
    @_cached
    def num_pixels(self):
        return np.sum(self.image)

    @property
    @_cached
    def area(self):
        return np.sum(self.image) * self._pixel_area

    @property
    def bbox(self):
        """
        Returns
        -------
        A tuple of the bounding box's start coordinates for each dimension,
        followed by the end coordinates for each dimension
        """
        return tuple([self.slice[i].start for i in range(self._ndim)] + [self.slice[i].stop for i in range(self._ndim)])

    @property
    def area_bbox(self):
        return self.image.size * self._pixel_area

    @property
    def centroid(self):
        return tuple(self.coords_scaled.mean(axis=0))

    @property
    @_cached
    def area_convex(self):
        return np.sum(self.image_convex) * self._pixel_area

    @property
    @_cached
    def image_convex(self):
        from ..morphology.convex_hull import convex_hull_image
        return convex_hull_image(self.image)

    @property
    def coords_scaled(self):
        indices = np.argwhere(self.image)
        object_offset = np.array([self.slice[i].start for i in range(self._ndim)])
        return (object_offset + indices) * self._spacing + self._offset

    @property
    def coords(self):
        indices = np.argwhere(self.image)
        object_offset = np.array([self.slice[i].start for i in range(self._ndim)])
        return object_offset + indices + self._offset

    @property
    @only2d
    def eccentricity(self):
        l1, l2 = self.inertia_tensor_eigvals
        if l1 == 0:
            return 0
        return sqrt(1 - l2 / l1)

    @property
    def equivalent_diameter_area(self):
        return (2 * self._ndim * self.area / PI) ** (1 / self._ndim)

    @property
    def euler_number(self):
        if self._ndim not in [2, 3]:
            raise NotImplementedError('Euler number is implemented for 2D or 3D images only')
        return euler_number(self.image, self._ndim)

    @property
    def extent(self):
        return self.area / self.area_bbox

    @property
    def feret_diameter_max(self):
        identity_convex_hull = np.pad(self.image_convex, 2, mode='constant', constant_values=0)
        if self._ndim == 2:
            coordinates = np.vstack(find_contours(identity_convex_hull, 0.5, fully_connected='high'))
        elif self._ndim == 3:
            coordinates, _, _, _ = marching_cubes(identity_convex_hull, level=0.5)
        distances = pdist(coordinates * self._spacing, 'sqeuclidean')
        return sqrt(np.max(distances))

    @property
    def area_filled(self):
        return np.sum(self.image_filled) * self._pixel_area

    @property
    @_cached
    def image_filled(self):
        structure = np.ones((3,) * self._ndim)
        return ndi.binary_fill_holes(self.image, structure)

    @property
    @_cached
    def image(self):
        return self._label_image[self.slice] == self.label

    @property
    @_cached
    def inertia_tensor(self):
        mu = self.moments_central
        return _moments.inertia_tensor(self.image, mu, spacing=self._spacing)

    @property
    @_cached
    def inertia_tensor_eigvals(self):
        return _moments.inertia_tensor_eigvals(self.image, T=self.inertia_tensor)

    @property
    @_cached
    def image_intensity(self):
        if self._intensity_image is None:
            raise AttributeError('No intensity image specified.')
        image = self.image if not self._multichannel else np.expand_dims(self.image, self._ndim)
        return self._intensity_image[self.slice] * image

    def _image_intensity_double(self):
        return self.image_intensity.astype(np.float64, copy=False)

    @property
    def centroid_local(self):
        M = self.moments
        M0 = M[(0,) * self._ndim]

        def _get_element(axis):
            return (0,) * axis + (1,) + (0,) * (self._ndim - 1 - axis)
        return np.asarray(tuple((M[_get_element(axis)] / M0 for axis in range(self._ndim))))

    @property
    def intensity_max(self):
        vals = self.image_intensity[self.image]
        return np.max(vals, axis=0).astype(np.float64, copy=False)

    @property
    def intensity_mean(self):
        return np.mean(self.image_intensity[self.image], axis=0)

    @property
    def intensity_min(self):
        vals = self.image_intensity[self.image]
        return np.min(vals, axis=0).astype(np.float64, copy=False)

    @property
    def intensity_std(self):
        vals = self.image_intensity[self.image]
        return np.std(vals, axis=0)

    @property
    def axis_major_length(self):
        if self._ndim == 2:
            l1 = self.inertia_tensor_eigvals[0]
            return 4 * sqrt(l1)
        elif self._ndim == 3:
            ev = self.inertia_tensor_eigvals
            return sqrt(10 * (ev[0] + ev[1] - ev[2]))
        else:
            raise ValueError('axis_major_length only available in 2D and 3D')

    @property
    def axis_minor_length(self):
        if self._ndim == 2:
            l2 = self.inertia_tensor_eigvals[-1]
            return 4 * sqrt(l2)
        elif self._ndim == 3:
            ev = self.inertia_tensor_eigvals
            return sqrt(10 * (-ev[0] + ev[1] + ev[2]))
        else:
            raise ValueError('axis_minor_length only available in 2D and 3D')

    @property
    @_cached
    def moments(self):
        M = _moments.moments(self.image.astype(np.uint8), 3, spacing=self._spacing)
        return M

    @property
    @_cached
    def moments_central(self):
        mu = _moments.moments_central(self.image.astype(np.uint8), self.centroid_local, order=3, spacing=self._spacing)
        return mu

    @property
    @only2d
    def moments_hu(self):
        if any((s != 1.0 for s in self._spacing)):
            raise NotImplementedError('`moments_hu` supports spacing = (1, 1) only')
        return _moments.moments_hu(self.moments_normalized)

    @property
    @_cached
    def moments_normalized(self):
        return _moments.moments_normalized(self.moments_central, 3, spacing=self._spacing)

    @property
    @only2d
    def orientation(self):
        a, b, b, c = self.inertia_tensor.flat
        if a - c == 0:
            if b < 0:
                return PI / 4.0
            else:
                return -PI / 4.0
        else:
            return 0.5 * atan2(-2 * b, c - a)

    @property
    @only2d
    def perimeter(self):
        if len(np.unique(self._spacing)) != 1:
            raise NotImplementedError('`perimeter` supports isotropic spacings only')
        return perimeter(self.image, 4) * self._spacing[0]

    @property
    @only2d
    def perimeter_crofton(self):
        if len(np.unique(self._spacing)) != 1:
            raise NotImplementedError('`perimeter` supports isotropic spacings only')
        return perimeter_crofton(self.image, 4) * self._spacing[0]

    @property
    def solidity(self):
        return self.area / self.area_convex

    @property
    def centroid_weighted(self):
        ctr = self.centroid_weighted_local
        return tuple((idx + slc.start * spc for idx, slc, spc in zip(ctr, self.slice, self._spacing)))

    @property
    def centroid_weighted_local(self):
        M = self.moments_weighted
        M0 = M[(0,) * self._ndim]

        def _get_element(axis):
            return (0,) * axis + (1,) + (0,) * (self._ndim - 1 - axis)
        return np.asarray(tuple((M[_get_element(axis)] / M0 for axis in range(self._ndim))))

    @property
    @_cached
    def moments_weighted(self):
        image = self._image_intensity_double()
        if self._multichannel:
            moments = np.stack([_moments.moments(image[..., i], order=3, spacing=self._spacing) for i in range(image.shape[-1])], axis=-1)
        else:
            moments = _moments.moments(image, order=3, spacing=self._spacing)
        return moments

    @property
    @_cached
    def moments_weighted_central(self):
        ctr = self.centroid_weighted_local
        image = self._image_intensity_double()
        if self._multichannel:
            moments_list = [_moments.moments_central(image[..., i], center=ctr[..., i], order=3, spacing=self._spacing) for i in range(image.shape[-1])]
            moments = np.stack(moments_list, axis=-1)
        else:
            moments = _moments.moments_central(image, ctr, order=3, spacing=self._spacing)
        return moments

    @property
    @only2d
    def moments_weighted_hu(self):
        if not (np.array(self._spacing) == np.array([1, 1])).all():
            raise NotImplementedError('`moments_hu` supports spacing = (1, 1) only')
        nu = self.moments_weighted_normalized
        if self._multichannel:
            nchannels = self._intensity_image.shape[-1]
            return np.stack([_moments.moments_hu(nu[..., i]) for i in range(nchannels)], axis=-1)
        else:
            return _moments.moments_hu(nu)

    @property
    @_cached
    def moments_weighted_normalized(self):
        mu = self.moments_weighted_central
        if self._multichannel:
            nchannels = self._intensity_image.shape[-1]
            return np.stack([_moments.moments_normalized(mu[..., i], order=3, spacing=self._spacing) for i in range(nchannels)], axis=-1)
        else:
            return _moments.moments_normalized(mu, order=3, spacing=self._spacing)

    def __iter__(self):
        props = PROP_VALS
        if self._intensity_image is None:
            unavailable_props = _require_intensity_image
            props = props.difference(unavailable_props)
        return iter(sorted(props))

    def __getitem__(self, key):
        value = getattr(self, key, None)
        if value is not None:
            return value
        else:
            return getattr(self, PROPS[key])

    def __eq__(self, other):
        if not isinstance(other, RegionProperties):
            return False
        for key in PROP_VALS:
            try:
                np.testing.assert_equal(getattr(self, key, None), getattr(other, key, None))
            except AssertionError:
                return False
        return True