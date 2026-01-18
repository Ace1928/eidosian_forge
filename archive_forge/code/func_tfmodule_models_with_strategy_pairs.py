import os
import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
from keras.src.distribute import model_combinations
def tfmodule_models_with_strategy_pairs():
    return tf.__internal__.test.combinations.combine(model_and_input=[model_combinations.simple_tfmodule_model], distribution_for_saving=strategies, distribution_for_restoring=strategies, mode=['eager'])