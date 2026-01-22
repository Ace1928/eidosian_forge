import argparse
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
import yaml
from transformers import (
Convert FastSpeech2Conformer checkpoint.