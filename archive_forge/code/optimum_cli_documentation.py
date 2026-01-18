import importlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union
from ..utils import logging
from .base import BaseOptimumCLICommand, CommandInfo, RootOptimumCLICommand
from .env import EnvironmentCommand
from .export import ExportCommand
from .onnxruntime import ONNXRuntimeCommand

    Registers a command as being a subcommand of `parent_command`.

    Args:
        command_or_command_info (`Union[Type[BaseOptimumCLICommand], CommandInfo]`):
            The command to register.
        parent_command (`BaseOptimumCLICommand`):
            The instance of the parent command.
    