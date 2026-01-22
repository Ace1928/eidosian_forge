import multiprocessing.dummy
import queue
import random
import threading
import time
import warnings
import weakref
from contextlib import closing
import numpy as np
import tree
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
def _make_multiprocessed_generator_fn(self):
    workers = self.py_dataset.workers
    use_multiprocessing = self.py_dataset.use_multiprocessing
    if workers > 1 or (workers > 0 and use_multiprocessing):

        def generator_fn():
            self.enqueuer = OrderedEnqueuer(self.py_dataset, use_multiprocessing=use_multiprocessing, shuffle=self.shuffle)
            self.enqueuer.start(workers=workers, max_queue_size=self.py_dataset.max_queue_size)
            return self.enqueuer.get()
    else:

        def generator_fn():
            order = range(len(self.py_dataset))
            if self.shuffle:
                order = list(order)
                random.shuffle(order)
            for i in order:
                yield self.py_dataset[i]
    return generator_fn