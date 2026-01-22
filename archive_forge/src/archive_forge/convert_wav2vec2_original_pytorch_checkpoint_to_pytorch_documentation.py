import argparse
import json
import os
import fairseq
import torch
from fairseq.data import Dictionary
from transformers import (
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForSequenceClassification

    Copy/paste/tweak model's weights to transformers design.
    