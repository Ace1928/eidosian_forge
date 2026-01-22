from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from tqdm.auto import tqdm
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ops
import wandb
Plot validation results in a table.