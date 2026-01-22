import argparse
import copy
import json
import os
import re
import sys
import yaml
import wandb
from wandb import trigger
from wandb.util import add_import_hook, get_optional_module
def _monkey_tfkeras():
    from tensorflow import keras as tfkeras
    from wandb.integration.keras import WandbCallback
    models = getattr(tfkeras, 'models', None)
    if not models:
        return
    models.Model._keras_or_tfkeras = tfkeras
    if models.Model.fit == _magic_fit:
        return
    models.Model._fit = models.Model.fit
    models.Model.fit = _magic_fit
    models.Model._fit_generator = models.Model.fit_generator
    models.Model.fit_generator = _magic_fit_generator