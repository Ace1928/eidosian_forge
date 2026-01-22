import argparse
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, RobertaPreLayerNormConfig, RobertaPreLayerNormForMaskedLM
from transformers.utils import logging

    Copy/paste/tweak roberta_prelayernorm's weights to our BERT structure.
    