import argparse
import os
import torch
from huggingface_hub import hf_hub_download
from transformers import ClvpConfig, ClvpModelForConditionalGeneration
def update_index(present_index):
    if present_index % 2 == 0:
        return int(present_index / 2)
    else:
        return int((present_index - 1) / 2)