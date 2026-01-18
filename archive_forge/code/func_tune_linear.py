import argparse
from torch_regression_example import get_datasets, train_func
import ray
from ray import tune
from ray.train import DataConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune.tune_config import TuneConfig
from ray.tune.tuner import Tuner
def tune_linear(num_workers, num_samples, use_gpu):
    train_dataset, val_dataset = get_datasets()
    config = {'lr': 0.01, 'hidden_size': 1, 'batch_size': 4, 'epochs': 3}
    trainer = TorchTrainer(train_loop_per_worker=train_func, train_loop_config=config, scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu), datasets={'train': train_dataset, 'validation': val_dataset}, dataset_config=DataConfig(datasets_to_split=['train']))
    tuner = Tuner(trainer, param_space={'train_loop_config': {'lr': tune.loguniform(0.0001, 0.1), 'batch_size': tune.choice([4, 16, 32]), 'epochs': 3}}, tune_config=TuneConfig(num_samples=num_samples, metric='loss', mode='min'))
    result_grid = tuner.fit()
    best_result = result_grid.get_best_result()
    print(best_result)
    return best_result