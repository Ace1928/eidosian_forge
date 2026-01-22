import logging
import operator
import os
import shutil
import sys
from itertools import chain
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K  # noqa: N812
import wandb
from wandb.sdk.integration_utils.data_logging import ValidationDataLogger
from wandb.sdk.lib.deprecate import Deprecated, deprecate
from wandb.util import add_import_hook
def _logits_to_captions(self, logits):
    if logits[0].shape[-1] == 1:
        if len(self.labels) == 2:
            captions = [self.labels[1] if logits[0] > 0.5 else self.labels[0] for logit in logits]
        else:
            if len(self.labels) != 0:
                wandb.termwarn('keras model is producing a single output, so labels should be a length two array: ["False label", "True label"].')
            captions = [logit[0] for logit in logits]
    else:
        labels = np.argmax(np.stack(logits), axis=1)
        if len(self.labels) > 0:
            captions = []
            for label in labels:
                try:
                    captions.append(self.labels[label])
                except IndexError:
                    captions.append(label)
        else:
            captions = labels
    return captions