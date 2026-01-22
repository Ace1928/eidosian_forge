import argparse
import datetime
import json
import os
import re
from pathlib import Path
from typing import Tuple
import yaml
from tqdm import tqdm
from transformers.models.marian.convert_marian_to_pytorch import (

        Construct card from data parsed from YAML and the model's name. upload command: aws s3 sync model_card_dir
        s3://models.huggingface.co/bert/Helsinki-NLP/ --dryrun
        