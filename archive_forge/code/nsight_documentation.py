import os
import sys
import logging
import asyncio
import subprocess
import copy
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.plugin import RuntimeEnvPlugin
from ray._private.utils import (
from ray.exceptions import RuntimeEnvSetupError

        Function to validate if nsight_config is a valid nsight profile options
        Args:
            nsight_config: dictionary mapping nsight option to it's value
        Returns:
            a tuple consists of a boolean indicating if the nsight_config
            is valid option and an error message if the nsight_config is invalid
        