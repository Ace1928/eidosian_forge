import argparse
import json
import os
import torch
from transformers import LukeConfig, LukeModel, LukeTokenizer, RobertaTokenizer
from transformers.tokenization_utils_base import AddedToken
Convert LUKE checkpoint.