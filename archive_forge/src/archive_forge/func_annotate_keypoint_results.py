from typing import Any, Optional
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from ultralytics.engine.results import Results
from ultralytics.models.yolo.pose import PosePredictor
from ultralytics.utils.plotting import Annotator
import wandb
from wandb.integration.ultralytics.bbox_utils import (
def annotate_keypoint_results(result: Results, visualize_skeleton: bool):
    annotator = Annotator(np.ascontiguousarray(result.orig_img[:, :, ::-1]))
    key_points = result.keypoints.data.numpy()
    for idx in range(key_points.shape[0]):
        annotator.kpts(key_points[idx], kpt_line=visualize_skeleton)
    return annotator.im