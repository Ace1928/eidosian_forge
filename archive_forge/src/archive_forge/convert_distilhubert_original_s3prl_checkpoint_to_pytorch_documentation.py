import argparse
import torch
from s3prl.hub import distilhubert
from transformers import HubertConfig, HubertModel, Wav2Vec2FeatureExtractor, logging

    Copy/paste/tweak model's weights to transformers design.
    