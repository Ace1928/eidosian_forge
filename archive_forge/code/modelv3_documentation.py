import numpy as np
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_tf, try_import_torch
Example of using the Keras functional API to define an RNN model.