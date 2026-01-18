import time
import numpy as np
import torch
import ray
import ray.train.torch
from ray import train, tune
from ray.train import ScalingConfig
from ray.train.horovod import HorovodTrainer
from ray.tune.tune_config import TuneConfig
from ray.tune.tuner import Tuner
def tune_horovod(num_workers, num_samples, use_gpu, mode='square', x_max=1.0):
    horovod_trainer = HorovodTrainer(train_loop_per_worker=train_loop_per_worker, scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu), train_loop_config={'mode': mode, 'x_max': x_max})
    tuner = Tuner(horovod_trainer, param_space={'train_loop_config': {'lr': tune.uniform(0.1, 1)}}, tune_config=TuneConfig(mode='min', metric='loss', num_samples=num_samples), _tuner_kwargs={'fail_fast': True})
    result_grid = tuner.fit()
    print('Best hyperparameters found were: ', result_grid.get_best_result().config)