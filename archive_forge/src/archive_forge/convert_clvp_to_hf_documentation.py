import argparse
import os
import torch
from huggingface_hub import hf_hub_download
from transformers import ClvpConfig, ClvpModelForConditionalGeneration

Weights conversion script for CLVP
