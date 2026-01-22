from typing import Any, Optional
import numpy as np
from tqdm.auto import tqdm
from ultralytics.engine.results import Results
from ultralytics.models.yolo.classify import ClassificationPredictor
import wandb
Plot classification results to a `wandb.Table`.