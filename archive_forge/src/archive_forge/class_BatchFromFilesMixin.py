import collections
import multiprocessing
import os
import threading
import warnings
import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.trainers.data_adapters.py_dataset_adapter import PyDataset
from keras.src.utils import image_utils
from keras.src.utils import io_utils
from keras.src.utils.module_utils import scipy
class BatchFromFilesMixin:
    """Adds methods related to getting batches from filenames.

    It includes the logic to transform image files to batches.
    """

    def set_processing_attrs(self, image_data_generator, target_size, color_mode, data_format, save_to_dir, save_prefix, save_format, subset, interpolation, keep_aspect_ratio):
        """Sets attributes to use later for processing files into a batch.

        Args:
            image_data_generator: Instance of `ImageDataGenerator`
                to use for random transformations and normalization.
            target_size: tuple of integers, dimensions to resize input images
            to.
            color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`.
                Color mode to read images.
            data_format: String, one of `channels_first`, `channels_last`.
            save_to_dir: Optional directory where to save the pictures
                being yielded, in a viewable format. This is useful
                for visualizing the random transformations being
                applied, for debugging purposes.
            save_prefix: String prefix to use for saving sample
                images (if `save_to_dir` is set).
            save_format: Format to use for saving sample images
                (if `save_to_dir` is set).
            subset: Subset of data (`"training"` or `"validation"`) if
                validation_split is set in ImageDataGenerator.
            interpolation: Interpolation method used to resample the image if
                the target size is different from that of the loaded image.
                Supported methods are "nearest", "bilinear", and "bicubic". If
                PIL version 1.1.3 or newer is installed, "lanczos" is also
                supported. If PIL version 3.4.0 or newer is installed, "box" and
                "hamming" are also supported. By default, "nearest" is used.
            keep_aspect_ratio: Boolean, whether to resize images to a target
                size without aspect ratio distortion. The image is cropped in
                the center with target aspect ratio before resizing.
        """
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.keep_aspect_ratio = keep_aspect_ratio
        if color_mode not in {'rgb', 'rgba', 'grayscale'}:
            raise ValueError(f'Invalid color mode: {color_mode}; expected "rgb", "rgba", or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgba':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (4,)
            else:
                self.image_shape = (4,) + self.target_size
        elif self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        elif self.data_format == 'channels_last':
            self.image_shape = self.target_size + (1,)
        else:
            self.image_shape = (1,) + self.target_size
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation
        if subset is not None:
            validation_split = self.image_data_generator._validation_split
            if subset == 'validation':
                split = (0, validation_split)
            elif subset == 'training':
                split = (validation_split, 1)
            else:
                raise ValueError(f'Invalid subset name: {subset};expected "training" or "validation"')
        else:
            split = None
        self.split = split
        self.subset = subset

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        Args:
            index_array: Array of sample indices to include in batch.
        Returns:
            A batch of transformed samples.
        """
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=self.dtype)
        filepaths = self.filepaths
        for i, j in enumerate(index_array):
            img = image_utils.load_img(filepaths[j], color_mode=self.color_mode, target_size=self.target_size, interpolation=self.interpolation, keep_aspect_ratio=self.keep_aspect_ratio)
            x = image_utils.img_to_array(img, data_format=self.data_format)
            if hasattr(img, 'close'):
                img.close()
            if self.image_data_generator:
                params = self.image_data_generator.get_random_transform(x.shape)
                x = self.image_data_generator.apply_transform(x, params)
                x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = image_utils.array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix, index=j, hash=np.random.randint(10000000.0), format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode in {'binary', 'sparse'}:
            batch_y = np.empty(len(batch_x), dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i] = self.classes[n_observation]
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), len(self.class_indices)), dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i, self.classes[n_observation]] = 1.0
        elif self.class_mode == 'multi_output':
            batch_y = [output[index_array] for output in self.labels]
        elif self.class_mode == 'raw':
            batch_y = self.labels[index_array]
        else:
            return batch_x
        if self.sample_weight is None:
            return (batch_x, batch_y)
        else:
            return (batch_x, batch_y, self.sample_weight[index_array])

    @property
    def filepaths(self):
        """List of absolute paths to image files."""
        raise NotImplementedError('`filepaths` property method has not been implemented in {}.'.format(type(self).__name__))

    @property
    def labels(self):
        """Class labels of every observation."""
        raise NotImplementedError('`labels` property method has not been implemented in {}.'.format(type(self).__name__))

    @property
    def sample_weight(self):
        raise NotImplementedError('`sample_weight` property method has not been implemented in {}.'.format(type(self).__name__))