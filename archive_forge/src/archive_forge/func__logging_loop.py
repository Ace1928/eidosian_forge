import collections
import pickle
import threading
import time
import timeit
from absl import flags
from absl import logging
import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.distribute import values as values_lib  
from tensorflow.python.framework import composite_tensor  
from tensorflow.python.framework import tensor_conversion_registry  
def _logging_loop(self):
    """Loop being run in a separate thread."""
    last_log_try = timeit.default_timer()
    while not self.terminator.isSet():
        try:
            self._log()
        except Exception:
            logging.fatal('Logging failed.', exc_info=True)
        now = timeit.default_timer()
        elapsed = now - last_log_try
        last_log_try = now
        self.period = min(self.period_factor * self.period, self.max_period)
        self.terminator.wait(timeout=max(0, self.period - elapsed))