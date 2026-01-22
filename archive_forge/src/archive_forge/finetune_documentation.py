import argparse
import multiprocessing
import os
import torch
import transformers
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
from trl import SFTTrainer

    Prints the number of trainable parameters in the model.
    