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
@dataclass
@PublicAPI(stability='stable')
class ScalingConfig:
    """Configuration for scaling training.

    Args:
        trainer_resources: Resources to allocate for the trainer. If None is provided,
            will default to 1 CPU for most trainers.
        num_workers: The number of workers (Ray actors) to launch.
            Each worker will reserve 1 CPU by default. The number of CPUs
            reserved by each worker can be overridden with the
            ``resources_per_worker`` argument.
        use_gpu: If True, training will be done on GPUs (1 per worker).
            Defaults to False. The number of GPUs reserved by each
            worker can be overridden with the ``resources_per_worker``
            argument.
        resources_per_worker: If specified, the resources
            defined in this Dict is reserved for each worker.
            Define the ``"CPU"`` and ``"GPU"`` keys (case-sensitive) to
            override the number of CPU or GPUs used by each worker.
        placement_strategy: The placement strategy to use for the
            placement group of the Ray actors. See :ref:`Placement Group
            Strategies <pgroup-strategy>` for the possible options.

    Example:

        .. code-block:: python

            from ray.train import ScalingConfig
            scaling_config = ScalingConfig(
                # Number of distributed workers.
                num_workers=2,
                # Turn on/off GPU.
                use_gpu=True,
                # Specify resources used for trainer.
                trainer_resources={"CPU": 1},
                # Try to schedule workers on different nodes.
                placement_strategy="SPREAD",
            )

    """
    trainer_resources: Optional[Union[Dict, SampleRange]] = None
    num_workers: Optional[Union[int, SampleRange]] = None
    use_gpu: Union[bool, SampleRange] = False
    resources_per_worker: Optional[Union[Dict, SampleRange]] = None
    placement_strategy: Union[str, SampleRange] = 'PACK'

    def __post_init__(self):
        if self.resources_per_worker:
            if not self.use_gpu and self.num_gpus_per_worker > 0:
                raise ValueError('`use_gpu` is False but `GPU` was found in `resources_per_worker`. Either set `use_gpu` to True or remove `GPU` from `resources_per_worker.')
            if self.use_gpu and self.num_gpus_per_worker == 0:
                raise ValueError('`use_gpu` is True but `GPU` is set to 0 in `resources_per_worker`. Either set `use_gpu` to False or request a positive number of `GPU` in `resources_per_worker.')

    def __repr__(self):
        return _repr_dataclass(self)

    def _repr_html_(self) -> str:
        return make_table_html_repr(obj=self, title=type(self).__name__)

    def __eq__(self, o: 'ScalingConfig') -> bool:
        if not isinstance(o, type(self)):
            return False
        return self.as_placement_group_factory() == o.as_placement_group_factory()

    @property
    def _resources_per_worker_not_none(self):
        if self.resources_per_worker is None:
            if self.use_gpu:
                return {'GPU': 1}
            else:
                return {'CPU': 1}
        resources_per_worker = {k: v for k, v in self.resources_per_worker.items() if v != 0}
        resources_per_worker.setdefault('GPU', int(self.use_gpu))
        return resources_per_worker

    @property
    def _trainer_resources_not_none(self):
        if self.trainer_resources is None:
            if self.num_workers:
                try:
                    import google.colab
                    trainer_resources = 0
                except ImportError:
                    trainer_resources = 1
            else:
                trainer_resources = 1
            return {'CPU': trainer_resources}
        return {k: v for k, v in self.trainer_resources.items() if v != 0}

    @property
    def total_resources(self):
        """Map of total resources required for the trainer."""
        total_resource_map = defaultdict(float, self._trainer_resources_not_none)
        num_workers = self.num_workers or 0
        for k, value in self._resources_per_worker_not_none.items():
            total_resource_map[k] += value * num_workers
        return dict(total_resource_map)

    @property
    def num_cpus_per_worker(self):
        """The number of CPUs to set per worker."""
        return self._resources_per_worker_not_none.get('CPU', 0)

    @property
    def num_gpus_per_worker(self):
        """The number of GPUs to set per worker."""
        return self._resources_per_worker_not_none.get('GPU', 0)

    @property
    def additional_resources_per_worker(self):
        """Resources per worker, not including CPU or GPU resources."""
        return {k: v for k, v in self._resources_per_worker_not_none.items() if k not in ['CPU', 'GPU']}

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

    @classmethod
    def from_placement_group_factory(cls, pgf: 'PlacementGroupFactory') -> 'ScalingConfig':
        """Create a ScalingConfig from a Tune's PlacementGroupFactory"""
        if pgf.head_bundle_is_empty:
            trainer_resources = {}
            worker_bundles = pgf.bundles
        else:
            trainer_resources = pgf.bundles[0]
            worker_bundles = pgf.bundles[1:]
        use_gpu = False
        placement_strategy = pgf.strategy
        resources_per_worker = None
        num_workers = None
        if worker_bundles:
            first_bundle = worker_bundles[0]
            if not all((bundle == first_bundle for bundle in worker_bundles[1:])):
                raise ValueError('All worker bundles (any other than the first one) must be equal to each other.')
            use_gpu = bool(first_bundle.get('GPU'))
            num_workers = len(worker_bundles)
            resources_per_worker = first_bundle
        return ScalingConfig(trainer_resources=trainer_resources, num_workers=num_workers, use_gpu=use_gpu, resources_per_worker=resources_per_worker, placement_strategy=placement_strategy)