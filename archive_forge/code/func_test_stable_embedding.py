from itertools import product
import math
import random
import time
import einops
import numpy as np
import pytest
from scipy.stats import norm
import torch
import bitsandbytes as bnb
from bitsandbytes import functional as F
from tests.helpers import (
def test_stable_embedding():
    layer = bnb.nn.StableEmbedding(1024, 1024)
    layer.reset_parameters()