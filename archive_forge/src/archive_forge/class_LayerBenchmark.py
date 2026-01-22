import itertools
import math
import random
import string
import time
import numpy as np
import tensorflow.compat.v2 as tf
import keras.src as keras
class LayerBenchmark(tf.test.Benchmark):
    """Benchmark the layer forward pass."""

    def report(self, name, keras_time, fc_time, iters):
        """Calculate and report benchmark statistics."""
        extras = {'fc_avg_time': fc_time, 'fc_vs_keras_sec': fc_time - keras_time, 'fc_vs_keras_pct': (fc_time - keras_time) / fc_time * 100, 'keras_faster_ratio': fc_time / keras_time}
        self.report_benchmark(iters=iters, wall_time=keras_time, extras=extras, name=name)