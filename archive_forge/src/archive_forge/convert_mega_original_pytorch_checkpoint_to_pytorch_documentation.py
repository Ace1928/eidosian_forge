import argparse
import os
import pickle as pkl
import torch
from torch import nn
from transformers import AutoTokenizer, MegaConfig, MegaForMaskedLM

        Perform a forward pass through the Mega encoder and the masked LM head. Returns logits for each vocabulary
        entry.

        If `batch_first` (default to align with Hugging Face tokenizer behavior), output will have the shape (Batch
        size, Sequence length, Vocab size); otherwise (S, B, V)
        