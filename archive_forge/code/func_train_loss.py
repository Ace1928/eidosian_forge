from gymnasium.spaces import Box, Discrete
import numpy as np
from rllib.models.tf.attention_net import TrXLNet
from ray.rllib.utils.framework import try_import_tf
def train_loss(targets, outputs):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=outputs)
    return tf.reduce_mean(loss)