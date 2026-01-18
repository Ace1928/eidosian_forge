import argparse
import os
import torch
import torch.utils.data
import torchvision
from filelock import FileLock
from torchvision import datasets, transforms
import ray
from ray import train
from ray.train import ScalingConfig
def trainer_init_per_worker(config):
    import composer.optim
    from composer.core.evaluator import Evaluator
    from composer.models.tasks import ComposerClassifier
    from torchmetrics.classification.accuracy import Accuracy
    BATCH_SIZE = 64
    model = ComposerClassifier(torchvision.models.resnet18(num_classes=10))
    mean = (0.507, 0.487, 0.441)
    std = (0.267, 0.256, 0.276)
    cifar10_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    data_directory = os.path.expanduser('~/data')
    with FileLock(os.path.expanduser('~/data.lock')):
        train_dataset = torch.utils.data.Subset(datasets.CIFAR10(data_directory, train=True, download=True, transform=cifar10_transforms), list(range(BATCH_SIZE * 10)))
        test_dataset = torch.utils.data.Subset(datasets.CIFAR10(data_directory, train=False, download=True, transform=cifar10_transforms), list(range(BATCH_SIZE * 10)))
    batch_size_per_worker = BATCH_SIZE // train.get_context().get_world_size()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_per_worker, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_per_worker, shuffle=True)
    train_dataloader = train.torch.prepare_data_loader(train_dataloader)
    test_dataloader = train.torch.prepare_data_loader(test_dataloader)
    evaluator = Evaluator(dataloader=test_dataloader, label='my_evaluator', metrics=Accuracy(task='multiclass', num_classes=10, top_k=1))
    optimizer = composer.optim.DecoupledSGDW(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.002)
    if config.pop('should_eval', False):
        config['eval_dataloader'] = evaluator
    return composer.trainer.Trainer(model=model, train_dataloader=train_dataloader, optimizers=optimizer, **config)