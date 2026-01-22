import argparse
import torch
import torchaudio
from fairseq2.data import Collater
from fairseq2.data.audio import WaveformToFbankConverter
from fairseq2.nn.padding import get_seqs_and_padding_mask
from seamless_communication.models.conformer_shaw import load_conformer_shaw_model
from transformers import (

    Copy/paste/tweak model's weights to transformers design.
    