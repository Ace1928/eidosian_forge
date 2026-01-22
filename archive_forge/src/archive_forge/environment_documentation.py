import os
import platform
import subprocess
import sys
from shutil import which
from typing import List
import torch

    Checks if all the current GPUs available support FP8.

    Notably must initialize `torch.cuda` to check.
    