import logging
from collections import defaultdict
from dataclasses import _MISSING_TYPE, dataclass, fields
from pathlib import Path
from typing import (
import pyarrow.fs
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.util.annotations import PublicAPI, Deprecated
from ray.widgets import Template, make_table_html_repr
from ray.data.preprocessor import Preprocessor
def as_placement_group_factory(self) -> 'PlacementGroupFactory':
    """Returns a PlacementGroupFactory to specify resources for Tune."""
    from ray.tune.execution.placement_groups import PlacementGroupFactory
    trainer_resources = self._trainer_resources_not_none
    trainer_bundle = [trainer_resources]
    worker_resources = {'CPU': self.num_cpus_per_worker, 'GPU': self.num_gpus_per_worker}
    worker_resources_extra = {} if self.resources_per_worker is None else self.resources_per_worker
    worker_bundles = [{**worker_resources, **worker_resources_extra} for _ in range(self.num_workers if self.num_workers else 0)]
    bundles = trainer_bundle + worker_bundles
    return PlacementGroupFactory(bundles, strategy=self.placement_strategy)