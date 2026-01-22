import argparse
import gc
import os
import safetensors
import torch
from huggingface_hub import hf_hub_download
from transformers import PhiConfig, PhiForCausalLM

Weights conversion script for Phi

This script downloads both Phi-1 and Phi-1.5 checkpoints to "checkpoint_path" and then converts the weights to
HugfgingFace model's format and saves them in "pytorch_dump_folder_path".

Example : $python ./convert_phi_weights_to_hf.py --model_name "microsoft/phi-2" --pytorch_dump_folder ./dump_folder/ --checkpoint_path ./ckpt_path/
