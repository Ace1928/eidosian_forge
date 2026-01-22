import json
import logging
import random
import warnings
import numpy as np
from ..base import numeric_types
from .. import ndarray as nd
from ..ndarray._internal import _cvcopyMakeBorder as copyMakeBorder
from .. import io
from .image import RandomOrderAug, ColorJitterAug, LightingAug, ColorNormalizeAug
from .image import ResizeAug, ForceResizeAug, CastAug, HueJitterAug, RandomGrayAug
from .image import fixed_crop, ImageIter, Augmenter
from ..util import is_np_array
from .. import numpy as _mx_np  # pylint: disable=reimported
class DetRandomCropAug(DetAugmenter):
    """Random cropping with constraints

    Parameters
    ----------
    min_object_covered : float, default=0.1
        The cropped area of the image must contain at least this fraction of
        any bounding box supplied. The value of this parameter should be non-negative.
        In the case of 0, the cropped area does not need to overlap any of the
        bounding boxes supplied.
    min_eject_coverage : float, default=0.3
        The minimum coverage of cropped sample w.r.t its original size. With this
        constraint, objects that have marginal area after crop will be discarded.
    aspect_ratio_range : tuple of floats, default=(0.75, 1.33)
        The cropped area of the image must have an aspect ratio = width / height
        within this range.
    area_range : tuple of floats, default=(0.05, 1.0)
        The cropped area of the image must contain a fraction of the supplied
        image within in this range.
    max_attempts : int, default=50
        Number of attempts at generating a cropped/padded region of the image of the
        specified constraints. After max_attempts failures, return the original image.
    """

    def __init__(self, min_object_covered=0.1, aspect_ratio_range=(0.75, 1.33), area_range=(0.05, 1.0), min_eject_coverage=0.3, max_attempts=50):
        if not isinstance(aspect_ratio_range, (tuple, list)):
            assert isinstance(aspect_ratio_range, numeric_types)
            logging.info('Using fixed aspect ratio: %s in DetRandomCropAug', str(aspect_ratio_range))
            aspect_ratio_range = (aspect_ratio_range, aspect_ratio_range)
        if not isinstance(area_range, (tuple, list)):
            assert isinstance(area_range, numeric_types)
            logging.info('Using fixed area range: %s in DetRandomCropAug', area_range)
            area_range = (area_range, area_range)
        super(DetRandomCropAug, self).__init__(min_object_covered=min_object_covered, aspect_ratio_range=aspect_ratio_range, area_range=area_range, min_eject_coverage=min_eject_coverage, max_attempts=max_attempts)
        self.min_object_covered = min_object_covered
        self.min_eject_coverage = min_eject_coverage
        self.max_attempts = max_attempts
        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range
        self.enabled = False
        if area_range[1] <= 0 or area_range[0] > area_range[1]:
            warnings.warn('Skip DetRandomCropAug due to invalid area_range: %s', area_range)
        elif aspect_ratio_range[0] > aspect_ratio_range[1] or aspect_ratio_range[0] <= 0:
            warnings.warn('Skip DetRandomCropAug due to invalid aspect_ratio_range: %s', aspect_ratio_range)
        else:
            self.enabled = True

    def __call__(self, src, label):
        """Augmenter implementation body"""
        crop = self._random_crop_proposal(label, src.shape[0], src.shape[1])
        if crop:
            x, y, w, h, label = crop
            src = fixed_crop(src, x, y, w, h, None)
        return (src, label)

    def _calculate_areas(self, label):
        """Calculate areas for multiple labels"""
        heights = np.maximum(0, label[:, 3] - label[:, 1])
        widths = np.maximum(0, label[:, 2] - label[:, 0])
        return heights * widths

    def _intersect(self, label, xmin, ymin, xmax, ymax):
        """Calculate intersect areas, normalized."""
        left = np.maximum(label[:, 0], xmin)
        right = np.minimum(label[:, 2], xmax)
        top = np.maximum(label[:, 1], ymin)
        bot = np.minimum(label[:, 3], ymax)
        invalid = np.where(np.logical_or(left >= right, top >= bot))[0]
        out = label.copy()
        out[:, 0] = left
        out[:, 1] = top
        out[:, 2] = right
        out[:, 3] = bot
        out[invalid, :] = 0
        return out

    def _check_satisfy_constraints(self, label, xmin, ymin, xmax, ymax, width, height):
        """Check if constrains are satisfied"""
        if (xmax - xmin) * (ymax - ymin) < 2:
            return False
        x1 = float(xmin) / width
        y1 = float(ymin) / height
        x2 = float(xmax) / width
        y2 = float(ymax) / height
        object_areas = self._calculate_areas(label[:, 1:])
        valid_objects = np.where(object_areas * width * height > 2)[0]
        if valid_objects.size < 1:
            return False
        intersects = self._intersect(label[valid_objects, 1:], x1, y1, x2, y2)
        coverages = self._calculate_areas(intersects) / object_areas[valid_objects]
        coverages = coverages[np.where(coverages > 0)[0]]
        return coverages.size > 0 and np.amin(coverages) > self.min_object_covered

    def _update_labels(self, label, crop_box, height, width):
        """Convert labels according to crop box"""
        xmin = float(crop_box[0]) / width
        ymin = float(crop_box[1]) / height
        w = float(crop_box[2]) / width
        h = float(crop_box[3]) / height
        out = label.copy()
        out[:, (1, 3)] -= xmin
        out[:, (2, 4)] -= ymin
        out[:, (1, 3)] /= w
        out[:, (2, 4)] /= h
        out[:, 1:5] = np.maximum(0, out[:, 1:5])
        out[:, 1:5] = np.minimum(1, out[:, 1:5])
        coverage = self._calculate_areas(out[:, 1:]) * w * h / self._calculate_areas(label[:, 1:])
        valid = np.logical_and(out[:, 3] > out[:, 1], out[:, 4] > out[:, 2])
        valid = np.logical_and(valid, coverage > self.min_eject_coverage)
        valid = np.where(valid)[0]
        if valid.size < 1:
            return None
        out = out[valid, :]
        return out

    def _random_crop_proposal(self, label, height, width):
        """Propose cropping areas"""
        from math import sqrt
        if not self.enabled or height <= 0 or width <= 0:
            return ()
        min_area = self.area_range[0] * height * width
        max_area = self.area_range[1] * height * width
        for _ in range(self.max_attempts):
            ratio = random.uniform(*self.aspect_ratio_range)
            if ratio <= 0:
                continue
            h = int(round(sqrt(min_area / ratio)))
            max_h = int(round(sqrt(max_area / ratio)))
            if round(max_h * ratio) > width:
                max_h = int((width + 0.4999999) / ratio)
            if max_h > height:
                max_h = height
            if h > max_h:
                h = max_h
            if h < max_h:
                h = random.randint(h, max_h)
            w = int(round(h * ratio))
            assert w <= width
            area = w * h
            if area < min_area:
                h += 1
                w = int(round(h * ratio))
                area = w * h
            if area > max_area:
                h -= 1
                w = int(round(h * ratio))
                area = w * h
            if not (min_area <= area <= max_area and 0 <= w <= width and (0 <= h <= height)):
                continue
            y = random.randint(0, max(0, height - h))
            x = random.randint(0, max(0, width - w))
            if self._check_satisfy_constraints(label, x, y, x + w, y + h, width, height):
                new_label = self._update_labels(label, (x, y, w, h), height, width)
                if new_label is not None:
                    return (x, y, w, h, new_label)
        return ()