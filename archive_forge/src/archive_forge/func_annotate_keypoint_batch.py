from typing import Any, Optional
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from ultralytics.engine.results import Results
from ultralytics.models.yolo.pose import PosePredictor
from ultralytics.utils.plotting import Annotator
import wandb
from wandb.integration.ultralytics.bbox_utils import (
def annotate_keypoint_batch(image_path: str, keypoints: Any, visualize_skeleton: bool):
    original_image = None
    with Image.open(image_path) as original_image:
        original_image = np.ascontiguousarray(original_image)
        annotator = Annotator(original_image)
        annotator.kpts(keypoints.numpy(), kpt_line=visualize_skeleton)
        return annotator.im