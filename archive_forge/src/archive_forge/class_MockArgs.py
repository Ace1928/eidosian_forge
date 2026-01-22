from collections import abc as container_abcs, defaultdict
from copy import deepcopy
from itertools import chain
import torch
import bitsandbytes.functional as F
class MockArgs:

    def __init__(self, initial_data):
        for key in initial_data:
            setattr(self, key, initial_data[key])