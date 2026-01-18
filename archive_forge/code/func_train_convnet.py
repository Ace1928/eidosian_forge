import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from ray.tune.examples.mnist_pytorch import test_func, ConvNet, get_data_loaders
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import PopulationBasedTraining
def train_convnet(config):
    step = 0
    train_loader, test_loader = get_data_loaders()
    model = ConvNet()
    optimizer = optim.SGD(model.parameters(), lr=config.get('lr', 0.01), momentum=config.get('momentum', 0.9))
    if train.get_checkpoint():
        print('Loading from checkpoint.')
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            path = os.path.join(loaded_checkpoint_dir, 'checkpoint.pt')
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model'])
            step = checkpoint['step']
    while True:
        ray.tune.examples.mnist_pytorch.train_func(model, optimizer, train_loader)
        acc = test_func(model, test_loader)
        checkpoint = None
        if step % 5 == 0:
            os.makedirs('my_model', exist_ok=True)
            torch.save({'step': step, 'model': model.state_dict()}, 'my_model/checkpoint.pt')
            checkpoint = Checkpoint.from_directory('my_model')
        step += 1
        train.report({'mean_accuracy': acc}, checkpoint=checkpoint)