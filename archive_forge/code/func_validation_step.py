import math
import os
import torch
from filelock import FileLock
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray import train, tune
def validation_step(self, val_batch, batch_idx):
    x, y = val_batch
    logits = self.forward(x)
    loss = F.nll_loss(logits, y)
    acc = self.accuracy(logits, y)
    return {'val_loss': loss, 'val_accuracy': acc}