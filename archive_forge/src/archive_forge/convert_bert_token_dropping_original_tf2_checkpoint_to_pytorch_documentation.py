import argparse
import tensorflow as tf
import torch
from transformers import BertConfig, BertForMaskedLM
from transformers.models.bert.modeling_bert import (
from transformers.utils import logging

This script converts a lm-head checkpoint from the "Token Dropping" implementation into a PyTorch-compatible BERT
model. The official implementation of "Token Dropping" can be found in the TensorFlow Models repository:

https://github.com/tensorflow/models/tree/master/official/projects/token_dropping
