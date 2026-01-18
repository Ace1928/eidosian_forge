import itertools
from warnings import simplefilter
import numpy as np
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
import wandb
from wandb.sklearn import utils
def validate_labels(*args, **kwargs):
    assert False