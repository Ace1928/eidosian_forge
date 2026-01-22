import argparse
import torch
from transformers import FunnelBaseModel, FunnelConfig, FunnelModel, load_tf_weights_in_funnel
from transformers.utils import logging
Convert Funnel checkpoint.