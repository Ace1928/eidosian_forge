import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from ...exporters import TasksManager
from ...exporters.tflite import QuantizationApproach
from ..base import BaseOptimumCLICommand
Defines the command line for the export with TensorFlow Lite.