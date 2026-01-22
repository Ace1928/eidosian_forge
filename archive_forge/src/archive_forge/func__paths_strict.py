from __future__ import annotations
import copy
import json
import logging
import re
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union
import requests
import yaml
from langchain_core.pydantic_v1 import ValidationError
@property
def _paths_strict(self) -> Paths:
    if not self.paths:
        raise ValueError('No paths found in spec')
    return self.paths