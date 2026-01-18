import json
import operator
import os
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union
import torch
from lightning_utilities.core.imports import compare_version
from lightning_fabric.utilities.types import _PATH
Checks if we currently have a loss-spike.