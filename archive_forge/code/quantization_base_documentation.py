import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union
Overwrite this method in subclass to define how to quantize your model for quantization