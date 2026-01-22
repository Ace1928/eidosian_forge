from __future__ import annotations
import collections.abc
import contextlib
from collections import defaultdict
from copy import copy
import torch
from torchvision import datasets, tv_tensors
from torchvision.transforms.v2 import functional as F
class WrapperFactories(dict):

    def register(self, dataset_cls):

        def decorator(wrapper_factory):
            self[dataset_cls] = wrapper_factory
            return wrapper_factory
        return decorator