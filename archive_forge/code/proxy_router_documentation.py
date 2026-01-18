import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple
from ray.serve._private.common import (
from ray.serve._private.constants import (
from ray.serve.handle import RayServeHandle
Return the handle that matches with endpoint.

        Args:
            target_app_name: app_name to match against.
        Returns:
            (route, handle, app_name, is_cross_language) for the single app if there
            is only one, else find the app and handle for exact match. Else return None.
        