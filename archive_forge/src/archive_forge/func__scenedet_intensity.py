import numpy as np
import scipy.ndimage
import scipy.spatial
from ..motion.gme import globalEdgeMotion
from ..utils import *
def _scenedet_intensity(videodata, parameter1, min_scene_len=2, colorspace='hsv'):
    detected_scenes = [0]
    numFrames, height, width, channels = videodata.shape
    for t in range(0, numFrames - 1):
        frame0 = videodata[t].astype(np.float32)
        frame1 = videodata[t + 1].astype(np.float32)
        delta = np.sum(np.abs(frame1 - frame0) / (height * width * channels))
        if delta > parameter1 and t - detected_scenes[len(detected_scenes) - 1] > min_scene_len:
            detected_scenes.append(t + 1)
    return np.array(detected_scenes)