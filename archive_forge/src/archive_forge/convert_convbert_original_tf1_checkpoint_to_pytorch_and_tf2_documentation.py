import argparse
from transformers import ConvBertConfig, ConvBertModel, TFConvBertModel, load_tf_weights_in_convbert
from transformers.utils import logging
Convert ConvBERT checkpoint.