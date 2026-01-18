import argparse
import ray
from ray import tune
from ray.train.examples.tf.tensorflow_mnist_example import train_func
from ray.train.tensorflow import TensorflowTrainer
from ray.tune.tune_config import TuneConfig
from ray.tune.tuner import Tuner
def tune_tensorflow_mnist(num_workers: int=2, num_samples: int=2, use_gpu: bool=False):
    scaling_config = dict(num_workers=num_workers, use_gpu=use_gpu)
    trainer = TensorflowTrainer(train_loop_per_worker=train_func, scaling_config=scaling_config)
    tuner = Tuner(trainer, tune_config=TuneConfig(num_samples=num_samples, metric='binary_crossentropy', mode='min'), param_space={'train_loop_config': {'lr': tune.loguniform(0.0001, 0.1), 'batch_size': tune.choice([32, 64, 128]), 'epochs': 3}})
    best_accuracy = tuner.fit().get_best_result().metrics['binary_crossentropy']
    print(f'Best accuracy config: {best_accuracy}')