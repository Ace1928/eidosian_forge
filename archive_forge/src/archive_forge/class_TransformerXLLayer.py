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
class TransformerXLLayer(TransformerLayer):

    def __init__(self, d_model, num_heads, d_ff, dropout_rate):
        super().__init__(d_model, num_heads, d_ff, dropout_rate)
        self.segment_level = SegmentLevelRecurrence(d_model)

    async def forward(self, x, memory=None):
        attention_output = await self.attention.forward(x, x, x, memory)
        x = self.layer_norm1.forward(x + attention_output)
        feed_forward_output = await self.feed_forward.forward(x)
        x = self.layer_norm2.forward(x + feed_forward_output)
        x, memory = self.segment_level.forward(x)
        return (x, memory)