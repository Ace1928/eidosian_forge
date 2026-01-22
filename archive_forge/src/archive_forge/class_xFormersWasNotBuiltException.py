import dataclasses
import json
import logging
import os
import platform
from typing import Any, Dict, Optional
import torch
class xFormersWasNotBuiltException(Exception):

    def __str__(self) -> str:
        return 'Need to compile C++ extensions to use all xFormers features.\n    Please install xformers properly (see https://github.com/facebookresearch/xformers#installing-xformers)\n' + UNAVAILABLE_FEATURES_MSG