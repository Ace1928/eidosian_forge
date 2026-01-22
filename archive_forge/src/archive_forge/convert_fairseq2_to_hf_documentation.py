import argparse
import os
from pathlib import Path
import torch
from accelerate.utils.modeling import find_tied_parameters
from seamless_communication.models.inference.translator import Translator
from transformers import (
from transformers.utils import logging

    Meta SeamlessM4T is made of 8 main components:
    - speech_encoder (#1) and speech_encoder_frontend (#2)
    - t2u_model (#3)
    - text_encoder (#4) and text_encoder_frontend (#5)
    - text_decoder (#6) [and text_decoder_frontend (#5) = equals to text_encoder_frontend]
    - final_proj (#7)
    - vocoder (#8)
    