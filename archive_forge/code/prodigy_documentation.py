from the local database to W&B in Tables format.
import wandb
from wandb.integration.prodigy import upload_dataset
import base64
import collections.abc
import io
import urllib
from copy import deepcopy
import pandas as pd
from PIL import Image
import wandb
from wandb import util
from wandb.plots.utils import test_missing
from wandb.sdk.lib import telemetry as wb_telemetry
Upload dataset from local database to Weights & Biases.

    Args:
        dataset_name: The name of the dataset in the Prodigy database.
    