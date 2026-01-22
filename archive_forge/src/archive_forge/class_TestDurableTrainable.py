from collections import Counter
import json
import numpy as np
import os
import pickle
import tempfile
import time
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.callback import Callback
from ray._private.test_utils import safe_write_to_results_json
class TestDurableTrainable(tune.Trainable):

    def __init__(self, *args, **kwargs):
        self.setup_env()
        super(TestDurableTrainable, self).__init__(*args, **kwargs)

    def setup_env(self):
        pass

    def setup(self, config):
        self._num_iters = int(config['num_iters'])
        self._sleep_time = config['sleep_time']
        self._score = config['score']
        self._checkpoint_iters = config['checkpoint_iters']
        self._checkpoint_size_b = config['checkpoint_size_b']
        self._checkpoint_num_items = self._checkpoint_size_b // 8
        self._iter = 0

    def step(self):
        if self._iter > 0:
            time.sleep(self._sleep_time)
        res = dict(score=self._iter + self._score)
        if self._iter >= self._num_iters:
            res['done'] = True
        self._iter += 1
        return res

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_file = os.path.join(tmp_checkpoint_dir, 'bogus.ckpt')
        checkpoint_data = np.random.uniform(0, 1, size=self._checkpoint_num_items)
        with open(checkpoint_file, 'wb') as fp:
            pickle.dump(checkpoint_data, fp)

    def load_checkpoint(self, checkpoint):
        pass