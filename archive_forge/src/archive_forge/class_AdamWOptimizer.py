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
class AdamWOptimizer(AdamOptimizer):

    def __init__(self, learning_rate, weight_decay, beta1=0.9, beta2=0.999, epsilon=1e-08):
        super().__init__(learning_rate, beta1, beta2, epsilon)
        self.weight_decay = weight_decay

    async def update_weights(self, model, gradients):
        for name, param in model.named_parameters():
            if param.requires_grad:
                grad = gradients[name]
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                exp_avg, exp_avg_sq = (state['exp_avg'], state['exp_avg_sq'])
                beta1, beta2 = (self.beta1, self.beta2)
                state['step'] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(self.epsilon)
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = self.learning_rate * math.sqrt(bias_correction2) / bias_correction1
                param.data.mul_(1 - self.weight_decay * self.learning_rate)
                param.data.addcdiv_(exp_avg, denom, value=-step_size)