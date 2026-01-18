from __future__ import annotations
import dataclasses
import datetime
import json
import os
import pathlib
from enum import Enum
import numpy as np
import pandas as pd
import pytest
import torch
from bson.objectid import ObjectId
from monty.json import MontyDecoder, MontyEncoder, MSONable, _load_redirect, jsanitize
from . import __version__ as tests_version
def test_redirect_settings_file(self):
    data = _load_redirect(os.path.join(test_dir, 'test_settings.yaml'))
    assert data == {'old_module': {'old_class': {'@class': 'new_class', '@module': 'new_module'}}}