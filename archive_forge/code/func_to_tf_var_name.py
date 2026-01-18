import argparse
import os
import numpy as np
import tensorflow as tf
import torch
from transformers import BertModel
def to_tf_var_name(name: str):
    for patt, repl in iter(var_map):
        name = name.replace(patt, repl)
    return f'bert/{name}'