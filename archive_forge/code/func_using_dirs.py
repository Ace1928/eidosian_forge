import os
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Iterator, List, Set, Tuple
import requests
import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor, nn
from transformers import CLIPTokenizer, DinatConfig, SwinConfig
from transformers.models.oneformer.image_processing_oneformer import OneFormerImageProcessor
from transformers.models.oneformer.modeling_oneformer import (
from transformers.models.oneformer.processing_oneformer import OneFormerProcessor
from transformers.utils import logging
@staticmethod
def using_dirs(checkpoints_dir: Path, config_dir: Path) -> Iterator[Tuple[object, Path, Path]]:
    checkpoints: List[Path] = checkpoints_dir.glob('**/*.pth')
    for checkpoint in checkpoints:
        logger.info(f'💪 Converting {checkpoint.stem}')
        config: Path = config_dir / f'{checkpoint.stem}.yaml'
        yield (config, checkpoint)