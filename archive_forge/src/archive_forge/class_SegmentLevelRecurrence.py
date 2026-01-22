import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
import re
import os
import json
from collections import defaultdict, deque
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import math
import tensorboard
from torch.utils.tensorboard import SummaryWriter
class SegmentLevelRecurrence:

    def __init__(self, d_model):
        self.d_model = d_model
        self.segment_weights = np.random.randn(d_model, d_model)
        self.segment_bias = np.zeros(d_model)

    def forward(self, x):
        segment_output = np.dot(x, self.segment_weights) + self.segment_bias
        x = x + segment_output
        return (x, segment_output)