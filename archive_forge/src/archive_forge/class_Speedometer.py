import logging
import math
import time
from .model import save_checkpoint
class Speedometer(object):
    """Logs training speed and evaluation metrics periodically.

    Parameters
    ----------
    batch_size: int
        Batch size of data.
    frequent: int
        Specifies how frequently training speed and evaluation metrics
        must be logged. Default behavior is to log once every 50 batches.
    auto_reset : bool
        Reset the evaluation metrics after each log.

    Example
    -------
    >>> # Print training speed and evaluation metrics every ten batches. Batch size is one.
    >>> module.fit(iterator, num_epoch=n_epoch,
    ... batch_end_callback=mx.callback.Speedometer(1, 10))
    Epoch[0] Batch [10] Speed: 1910.41 samples/sec  Train-accuracy=0.200000
    Epoch[0] Batch [20] Speed: 1764.83 samples/sec  Train-accuracy=0.400000
    Epoch[0] Batch [30] Speed: 1740.59 samples/sec  Train-accuracy=0.500000
    """

    def __init__(self, batch_size, frequent=50, auto_reset=True):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0
        self.auto_reset = auto_reset

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count
        if self.init:
            if count % self.frequent == 0:
                try:
                    speed = self.frequent * self.batch_size / (time.time() - self.tic)
                except ZeroDivisionError:
                    speed = float('inf')
                if param.eval_metric is not None:
                    name_value = param.eval_metric.get_name_value()
                    if self.auto_reset:
                        param.eval_metric.reset_local()
                        msg = 'Epoch[%d] Batch [%d-%d]\tSpeed: %.2f samples/sec'
                        msg += '\t%s=%f' * len(name_value)
                        logging.info(msg, param.epoch, count - self.frequent, count, speed, *sum(name_value, ()))
                    else:
                        msg = 'Epoch[%d] Batch [0-%d]\tSpeed: %.2f samples/sec'
                        msg += '\t%s=%f' * len(name_value)
                        logging.info(msg, param.epoch, count, speed, *sum(name_value, ()))
                else:
                    logging.info('Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec', param.epoch, count, speed)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()