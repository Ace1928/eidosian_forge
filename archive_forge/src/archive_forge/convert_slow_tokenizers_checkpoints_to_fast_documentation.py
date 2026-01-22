import argparse
import os
import transformers
from .convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS
from .utils import logging
 Convert slow tokenizers checkpoints in fast (serialization format of the `tokenizers` library)