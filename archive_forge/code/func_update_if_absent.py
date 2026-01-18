import logging
import os
from typing import Dict, List, Optional
import ray._private.ray_constants as ray_constants
from ray._private.utils import (
def update_if_absent(self, **kwargs):
    """Update the settings when the target fields are None.

        Args:
            kwargs: The keyword arguments to set corresponding fields.
        """
    for arg in kwargs:
        if hasattr(self, arg):
            if getattr(self, arg) is None:
                setattr(self, arg, kwargs[arg])
        else:
            raise ValueError(f'Invalid RayParams parameter in update_if_absent: {arg}')
    self._check_usage()