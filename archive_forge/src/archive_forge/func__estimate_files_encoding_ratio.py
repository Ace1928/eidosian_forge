import io
import logging
import time
from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple, Union
import numpy as np
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.util import _check_import
from ray.data.block import Block, BlockMetadata
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.data.datasource.file_meta_provider import DefaultFileMetadataProvider
from ray.util.annotations import DeveloperAPI
def _estimate_files_encoding_ratio(self) -> float:
    """Return an estimate of the image files encoding ratio."""
    start_time = time.perf_counter()
    non_empty_path_and_size = list(filter(lambda p: p[1] > 0, zip(self._paths(), self._file_sizes())))
    num_files = len(non_empty_path_and_size)
    if num_files == 0:
        logger.warn('All input image files are empty. Use on-disk file size to estimate images in-memory size.')
        return IMAGE_ENCODING_RATIO_ESTIMATE_DEFAULT
    if self.size is not None and self.mode is not None:
        if self.mode in ['1', 'L', 'P']:
            dimension = 1
        elif self.mode in ['RGB', 'YCbCr', 'LAB', 'HSV']:
            dimension = 3
        elif self.mode in ['RGBA', 'CMYK', 'I', 'F']:
            dimension = 4
        else:
            logger.warn(f'Found unknown image mode: {self.mode}.')
            return IMAGE_ENCODING_RATIO_ESTIMATE_DEFAULT
        height, width = self.size
        single_image_size = height * width * dimension
        total_estimated_size = single_image_size * num_files
        total_file_size = sum((p[1] for p in non_empty_path_and_size))
        ratio = total_estimated_size / total_file_size
    else:
        ratio = IMAGE_ENCODING_RATIO_ESTIMATE_DEFAULT
    sampling_duration = time.perf_counter() - start_time
    if sampling_duration > 5:
        logger.warn(f'Image input size estimation took {round(sampling_duration, 2)} seconds.')
    logger.debug(f'Estimated image encoding ratio from sampling is {ratio}.')
    return max(ratio, IMAGE_ENCODING_RATIO_ESTIMATE_LOWER_BOUND)