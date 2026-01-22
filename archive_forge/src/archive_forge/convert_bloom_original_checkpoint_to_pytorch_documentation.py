import argparse
import json
import os
import re
import torch
from transformers import BloomConfig, BloomModel
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME
from transformers.utils import logging
Convert Megatron-DeepSpeed TP/PP weights mapping in transformers PP only