import random
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
Example of using embedded eager execution in a custom model.

    This shows how to use tf.py_function() to execute a snippet of TF code
    in eager mode. Here the `self.forward_eager` method just prints out
    the intermediate tensor for debug purposes, but you can in general
    perform any TF eager operation in tf.py_function().
    