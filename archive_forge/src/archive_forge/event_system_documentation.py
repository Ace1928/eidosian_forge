from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union
from ray.autoscaler._private.cli_logger import cli_logger
Clears stored callable objects for event.

        Args:
            event: Event that has callable objects stored in map.
                See CreateClusterEvent for details on the available events.
        