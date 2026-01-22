import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import sys
import os
from importlib import import_module
from .tpot import TPOTClassifier, TPOTRegressor
from ._version import __version__
def _read_data_file(args):
    input_data = pd.read_csv(args.INPUT_FILE, sep=args.INPUT_SEPARATOR, dtype=np.float64)
    if args.TARGET_NAME not in input_data.columns.values:
        raise ValueError('The provided data file does not seem to have a target column. Please make sure to specify the target column using the -target parameter.')
    return input_data