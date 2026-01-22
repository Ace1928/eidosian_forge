import argparse
import pickle
import numpy as np
import torch
from torch import nn
from transformers import ReformerConfig, ReformerModelWithLMHead
from transformers.utils import logging
Convert Reformer checkpoint.