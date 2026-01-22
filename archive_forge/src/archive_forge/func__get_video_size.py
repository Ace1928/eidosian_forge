from collections import deque
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import gym.error
from gym import Env, logger
from gym.core import ActType, ObsType
from gym.error import DependencyNotInstalled
from gym.logger import deprecation
def _get_video_size(self, zoom: Optional[float]=None) -> Tuple[int, int]:
    rendered = self.env.render()
    if isinstance(rendered, List):
        rendered = rendered[-1]
    assert rendered is not None and isinstance(rendered, np.ndarray)
    video_size = (rendered.shape[1], rendered.shape[0])
    if zoom is not None:
        video_size = (int(video_size[0] * zoom), int(video_size[1] * zoom))
    return video_size