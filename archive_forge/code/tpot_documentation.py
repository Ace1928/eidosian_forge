from sklearn.model_selection import train_test_split
from sklearn.utils import _safe_indexing
import numpy as np
from .base import TPOTBase
from .config.classifier import classifier_config_dict
from .config.regressor import regressor_config_dict
Set the sample of data used to verify pipelines work with the passed data set.

        