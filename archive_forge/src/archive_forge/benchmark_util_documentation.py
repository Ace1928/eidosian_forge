import timeit
import numpy as np
from keras.src import callbacks
from keras.src.benchmarks import distribution_util
Run models and measure the performance.

    Args:
      model_fn: Model function to be benchmarked.
      x: Input data. See `x` in the `fit()` method of `keras.Model`.
      y: Target data. See `y` in the `fit()` method of `keras.Model`.
      epochs: Integer. Number of epochs to train the model.
        If unspecified, `epochs` will default to 2.
      batch_size: Integer. Number of samples per gradient update. If
        unspecified, `batch_size` will default to 32.
      run_iters: Integer. Number of iterations to run the performance
        measurement.  If unspecified, `run_iters` will default to 4.
      optimizer: String (name of optimizer) or optimizer instance. See
        `tf.keras.optimizers`.
      loss: String (name of objective function), objective function or
        `tf.keras.losses.Loss` instance. See `tf.keras.losses`.
      metrics: Lists of metrics to be evaluated by the model during training.
        See `metrics` in the `compile()` method of  `keras.Model`.
      verbose: 0, 1, 2. Verbosity mode. See `verbose` in the `fit()` method of
        `keras.Model`. If unspecified, `verbose` will default to 0.
      num_gpus: Number of GPUs to run the model.
      distribution_strategy: Distribution strategies. It could be
        `multi_worker_mirrored`, `one_device`, `mirrored`. If unspecified,
        `distribution_strategy` will default to 'off'. Note that, `TPU`
        and `parameter_server` are not supported yet.

    Returns:
      Performance summary, which contains build_time, compile_time,
      startup_time, avg_epoch_time, wall_time, exp_per_sec, epochs,
      distribution_strategy.

    Raise:
      ValueError: If `x` is none or if `optimizer` is not provided or
      if `loss` is not provided or if `num_gpus` is negative.
    