from typing import Any, List
from torch import Tensor
from typing_extensions import override
from lightning_fabric.plugins.collectives.collective import Collective
from lightning_fabric.utilities.types import CollectibleGroup
Support for collective operations on a single device (no-op).

    .. warning:: This is an :ref:`experimental <versioning:Experimental API>` feature which is still in development.

    