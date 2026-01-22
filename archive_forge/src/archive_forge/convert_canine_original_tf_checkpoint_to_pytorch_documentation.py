import argparse
from transformers import CanineConfig, CanineModel, CanineTokenizer, load_tf_weights_in_canine
from transformers.utils import logging
Convert CANINE checkpoint.