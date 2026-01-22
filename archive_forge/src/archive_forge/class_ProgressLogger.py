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
class ProgressLogger(object):
    """Helper class for performing periodic logging of the training progress."""

    def __init__(self, summary_writer=None, initial_period=0.1, period_factor=1.01, max_period=10.0, starting_step=0):
        """Constructs ProgressLogger.

    Args:
      summary_writer: Tensorflow summary writer to use.
      initial_period: Initial logging period in seconds
        (how often logging happens).
      period_factor: Factor by which logging period is
        multiplied after each iteration (exponential back-off).
      max_period: Maximal logging period in seconds
        (the end of exponential back-off).
      starting_step: Step from which to start the summary writer.
    """
        self.summary_writer = None
        self.last_log_time = None
        self.last_log_step = 0
        self.period = initial_period
        self.period_factor = period_factor
        self.max_period = max_period
        self.log_keys = []
        self.log_keys_set = set()
        self.step_cnt = tf.Variable(-1, dtype=tf.int64)
        self.ready_values = tf.Variable([-1.0], dtype=tf.float32, shape=tf.TensorShape(None))
        self.logger_thread = None
        self.logging_callback = None
        self.terminator = None
        self.reset(summary_writer, starting_step)

    def reset(self, summary_writer=None, starting_step=0):
        """Resets the progress logger.

    Args:
      summary_writer: Tensorflow summary writer to use.
      starting_step: Step from which to start the summary writer.
    """
        self.summary_writer = summary_writer
        self.step_cnt.assign(starting_step)
        self.ready_values.assign([-1.0])
        self.last_log_time = timeit.default_timer()
        self.last_log_step = starting_step

    def start(self, logging_callback=None):
        assert self.logger_thread is None
        self.logging_callback = logging_callback
        self.terminator = threading.Event()
        self.logger_thread = threading.Thread(target=self._logging_loop)
        self.logger_thread.start()

    def shutdown(self):
        assert self.logger_thread
        self.terminator.set()
        self.logger_thread.join()
        self.logger_thread = None

    def log_session(self):
        return []

    def log(self, session, name, value):
        if name not in self.log_keys_set:
            self.log_keys.append(name)
            self.log_keys_set.add(name)
        session.append(value)

    def log_session_from_dict(self, dic):
        session = self.log_session()
        for key in dic:
            self.log(session, key, dic[key])
        return session

    def step_end(self, session, strategy=None, step_increment=1):
        logs = []
        for value in session:
            if strategy:
                value = tf.reduce_mean(tf.cast(strategy.experimental_local_results(value)[0], tf.float32))
            logs.append(value)
        self.ready_values.assign(logs)
        self.step_cnt.assign_add(step_increment)

    def _log(self):
        """Perform single round of logging."""
        logging_time = timeit.default_timer()
        step_cnt = self.step_cnt.read_value()
        if step_cnt == self.last_log_step:
            return
        values = self.ready_values.read_value().numpy()
        if values[0] == -1:
            return
        assert len(values) == len(self.log_keys), 'Mismatch between number of keys and values to log: %r vs %r' % (values, self.log_keys)
        if self.summary_writer:
            self.summary_writer.set_as_default()
        tf.summary.experimental.set_step(step_cnt.numpy())
        if self.logging_callback:
            self.logging_callback()
        for key, value in zip(self.log_keys, values):
            tf.summary.scalar(key, value)
        dt = logging_time - self.last_log_time
        df = tf.cast(step_cnt - self.last_log_step, tf.float32)
        tf.summary.scalar('speed/steps_per_sec', df / dt)
        self.last_log_time, self.last_log_step = (logging_time, step_cnt)

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