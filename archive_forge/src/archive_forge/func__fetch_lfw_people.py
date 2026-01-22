import logging
from numbers import Integral, Real
from os import PathLike, listdir, makedirs, remove
from os.path import exists, isdir, join
import numpy as np
from joblib import Memory
from ..utils import Bunch
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ._base import (
def _fetch_lfw_people(data_folder_path, slice_=None, color=False, resize=None, min_faces_per_person=0):
    """Perform the actual data loading for the lfw people dataset

    This operation is meant to be cached by a joblib wrapper.
    """
    person_names, file_paths = ([], [])
    for person_name in sorted(listdir(data_folder_path)):
        folder_path = join(data_folder_path, person_name)
        if not isdir(folder_path):
            continue
        paths = [join(folder_path, f) for f in sorted(listdir(folder_path))]
        n_pictures = len(paths)
        if n_pictures >= min_faces_per_person:
            person_name = person_name.replace('_', ' ')
            person_names.extend([person_name] * n_pictures)
            file_paths.extend(paths)
    n_faces = len(file_paths)
    if n_faces == 0:
        raise ValueError('min_faces_per_person=%d is too restrictive' % min_faces_per_person)
    target_names = np.unique(person_names)
    target = np.searchsorted(target_names, person_names)
    faces = _load_imgs(file_paths, slice_, color, resize)
    indices = np.arange(n_faces)
    np.random.RandomState(42).shuffle(indices)
    faces, target = (faces[indices], target[indices])
    return (faces, target, target_names)