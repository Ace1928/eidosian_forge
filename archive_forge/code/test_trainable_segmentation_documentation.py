from functools import partial
import numpy as np
import pytest
from scipy import spatial
from skimage.future import fit_segmenter, predict_segmenter, TrainableSegmenter
from skimage.feature import multiscale_basic_features
Test the object-oriented interface using the TrainableSegmenter class.