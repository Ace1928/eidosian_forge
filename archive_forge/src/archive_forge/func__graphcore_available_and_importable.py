import functools
import sys
from lightning_utilities.core.imports import RequirementCache, package_available
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def _graphcore_available_and_importable() -> bool:
    return bool(_LIGHTNING_GRAPHCORE_AVAILABLE) and _try_import_module('lightning_graphcore')