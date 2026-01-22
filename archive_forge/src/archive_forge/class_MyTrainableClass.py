import json
import time
import os
import numpy as np
import ray
from ray import train, tune
from ray.tune import Trainable
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
class MyTrainableClass(Trainable):
    """Example agent whose learning curve is a random sigmoid.

    The dummy hyperparameters "width" and "height" determine the slope and
    maximum reward value reached.
    """

    def setup(self, config):
        self.timestep = 0

    def step(self):
        self.timestep += 1
        v = np.tanh(float(self.timestep) / self.config.get('width', 1))
        v *= self.config.get('height', 1)
        time.sleep(0.1)
        return {'episode_reward_mean': v}

    def save_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, 'checkpoint')
        with open(path, 'w') as f:
            f.write(json.dumps({'timestep': self.timestep}))

    def load_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, 'checkpoint')
        with open(path, 'r') as f:
            self.timestep = json.loads(f.read())['timestep']