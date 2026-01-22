import abc
import logging
from typing import Tuple
import numpy as np
from ray.rllib.core.models.base import Model
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.models.specs.checker import (
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.util import log_once
class TfModel(Model, tf.keras.Model, abc.ABC):
    """Base class for RLlib's TensorFlow models.

    This class defines the interface for RLlib's TensorFlow models and checks
    whether inputs and outputs of __call__ are checked with `check_input_specs()` and
    `check_output_specs()` respectively.
    """

    def __init__(self, config: ModelConfig):
        tf.keras.Model.__init__(self)
        Model.__init__(self, config)
        if not is_input_decorated(self.call):
            raise ValueError(f"`{type(self).__name__}.call()` not decorated with input specification. Decorate it with @check_input_specs() to define a specification and resolve this Error. If you don't want to check anything, you can use an empty spec.")
        if is_output_decorated(self.call):
            if log_once('tf_model_forward_output_decorated'):
                logger.warning(f'`{type(self).__name__}.call()` decorated with output specification. This is not recommended because it can lead to slower execution. Remove @check_output_specs() from the forward method to resolve this.')

    @check_input_specs('input_specs')
    def call(self, input_dict: dict, **kwargs) -> dict:
        """Returns the output of this model for the given input.

        This method only makes sure that we have a spec-checked _forward() method.

        Args:
            input_dict: The input tensors.
            **kwargs: Forward compatibility kwargs.

        Returns:
            dict: The output tensors.
        """
        if self.config.always_check_shapes:

            @check_input_specs('input_specs', only_check_on_retry=False)
            @check_output_specs('output_specs')
            def checked_forward(self, input_data, **kwargs):
                return self._forward(input_data, **kwargs)
            return checked_forward(self, input_dict, **kwargs)
        return self._forward(input_dict, **kwargs)

    @override(Model)
    def get_num_parameters(self) -> Tuple[int, int]:
        return (sum((int(np.prod(w.shape)) for w in self.trainable_weights)), sum((int(np.prod(w.shape)) for w in self.non_trainable_weights)))

    @override(Model)
    def _set_to_dummy_weights(self, value_sequence=(-0.02, -0.01, 0.01, 0.02)):
        for i, w in enumerate(self.trainable_weights + self.non_trainable_weights):
            fill_val = value_sequence[i % len(value_sequence)]
            w.assign(tf.fill(w.shape, fill_val))