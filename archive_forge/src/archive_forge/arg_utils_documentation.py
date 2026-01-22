import argparse
import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
Shared CLI arguments for vLLM engine.