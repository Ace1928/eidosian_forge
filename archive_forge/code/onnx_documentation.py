import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING
from ...exporters import TasksManager
from ...utils import DEFAULT_DUMMY_SHAPES
from ..base import BaseOptimumCLICommand
Defines the command line for the export with ONNX.