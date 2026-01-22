import functools
import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
import keras.src as keras
from keras.src.distribute import distributed_training_utils
from keras.src.distribute.strategy_combinations import all_strategies
from keras.src.distribute.strategy_combinations import (
from keras.src.distribute.strategy_combinations import strategies_minus_tpu
from keras.src.mixed_precision import policy
from keras.src.utils import data_utils
class TestDistributionStrategyEmbeddingModelCorrectnessBase(TestDistributionStrategyCorrectnessBase):
    """Base class to test correctness of Keras models with embedding layers."""

    def get_data(self, count=_GLOBAL_BATCH_SIZE * _EVAL_STEPS, min_words=5, max_words=10, max_word_id=19, num_classes=2):
        distribution = []
        for _ in range(num_classes):
            dist = np.abs(np.random.randn(max_word_id))
            dist /= np.sum(dist)
            distribution.append(dist)
        features = []
        labels = []
        for _ in range(count):
            label = np.random.randint(0, num_classes, size=1)[0]
            num_words = np.random.randint(min_words, max_words, size=1)[0]
            word_ids = np.random.choice(max_word_id, size=num_words, replace=True, p=distribution[label])
            word_ids = word_ids
            labels.append(label)
            features.append(word_ids)
        features = data_utils.pad_sequences(features, maxlen=max_words)
        x_train = np.asarray(features, dtype=np.float32)
        y_train = np.asarray(labels, dtype=np.int32).reshape((count, 1))
        x_predict = x_train[:_GLOBAL_BATCH_SIZE]
        return (x_train, y_train, x_predict)