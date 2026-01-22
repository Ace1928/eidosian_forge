import argparse
import pytorch_lightning as pl
import torch
from torch import nn
from transformers import LongformerForQuestionAnswering, LongformerModel
Convert RoBERTa checkpoint.