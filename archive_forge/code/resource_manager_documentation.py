import abc
from typing import List, Optional
import ray
from ray.air.execution.resources.request import (
from ray.util.annotations import DeveloperAPI
We disallow serialization.

        Shared resource managers should live on an actor.
        