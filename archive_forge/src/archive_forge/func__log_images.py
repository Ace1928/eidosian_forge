import logging
import operator
import os
import shutil
import sys
from itertools import chain
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K  # noqa: N812
import wandb
from wandb.sdk.integration_utils.data_logging import ValidationDataLogger
from wandb.sdk.lib.deprecate import Deprecated, deprecate
from wandb.util import add_import_hook
def _log_images(self, num_images=36):
    validation_X = self.validation_data[0]
    validation_y = self.validation_data[1]
    validation_length = len(validation_X)
    if validation_length > num_images:
        indices = np.random.choice(validation_length, num_images, replace=False)
    else:
        indices = range(validation_length)
    test_data = []
    test_output = []
    for i in indices:
        test_example = validation_X[i]
        test_data.append(test_example)
        test_output.append(validation_y[i])
    if self.model.stateful:
        predictions = self.model.predict(np.stack(test_data), batch_size=1)
        self.model.reset_states()
    else:
        predictions = self.model.predict(np.stack(test_data), batch_size=self._prediction_batch_size)
        if len(predictions) != len(test_data):
            self._prediction_batch_size = 1
            predictions = self.model.predict(np.stack(test_data), batch_size=self._prediction_batch_size)
    if self.input_type == 'label':
        if self.output_type in ('image', 'images', 'segmentation_mask'):
            captions = self._logits_to_captions(test_data)
            output_image_data = self._masks_to_pixels(predictions) if self.output_type == 'segmentation_mask' else predictions
            reference_image_data = self._masks_to_pixels(test_output) if self.output_type == 'segmentation_mask' else test_output
            output_images = [wandb.Image(data, caption=captions[i], grouping=2) for i, data in enumerate(output_image_data)]
            reference_images = [wandb.Image(data, caption=captions[i]) for i, data in enumerate(reference_image_data)]
            return list(chain.from_iterable(zip(output_images, reference_images)))
    elif self.input_type in ('image', 'images', 'segmentation_mask'):
        input_image_data = self._masks_to_pixels(test_data) if self.input_type == 'segmentation_mask' else test_data
        if self.output_type == 'label':
            captions = self._logits_to_captions(predictions)
            return [wandb.Image(data, caption=captions[i]) for i, data in enumerate(test_data)]
        elif self.output_type in ('image', 'images', 'segmentation_mask'):
            output_image_data = self._masks_to_pixels(predictions) if self.output_type == 'segmentation_mask' else predictions
            reference_image_data = self._masks_to_pixels(test_output) if self.output_type == 'segmentation_mask' else test_output
            input_images = [wandb.Image(data, grouping=3) for i, data in enumerate(input_image_data)]
            output_images = [wandb.Image(data) for i, data in enumerate(output_image_data)]
            reference_images = [wandb.Image(data) for i, data in enumerate(reference_image_data)]
            return list(chain.from_iterable(zip(input_images, output_images, reference_images)))
        else:
            return [wandb.Image(img) for img in test_data]
    elif self.output_type in ('image', 'images', 'segmentation_mask'):
        output_image_data = self._masks_to_pixels(predictions) if self.output_type == 'segmentation_mask' else predictions
        reference_image_data = self._masks_to_pixels(test_output) if self.output_type == 'segmentation_mask' else test_output
        output_images = [wandb.Image(data, grouping=2) for i, data in enumerate(output_image_data)]
        reference_images = [wandb.Image(data) for i, data in enumerate(reference_image_data)]
        return list(chain.from_iterable(zip(output_images, reference_images)))