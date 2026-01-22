import argparse
import torch
from transformers import RoFormerConfig, RoFormerForMaskedLM, load_tf_weights_in_roformer
from transformers.utils import logging
Convert RoFormer checkpoint.