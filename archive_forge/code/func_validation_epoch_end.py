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
def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    avg_acc = torch.stack([x['val_accuracy'] for x in outputs]).mean()
    self.log('ptl/val_loss', avg_loss)
    self.log('ptl/val_accuracy', avg_acc)