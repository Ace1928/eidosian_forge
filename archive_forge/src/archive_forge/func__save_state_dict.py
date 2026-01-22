from typing import Optional
import warnings
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful
from .planner import SavePlanner
from .default_planner import DefaultSavePlanner
from .storage import (
from .metadata import Metadata, STATE_DICT_TYPE
from .utils import _DistWrapper
def _save_state_dict(state_dict: STATE_DICT_TYPE, storage_writer: StorageWriter, process_group: Optional[dist.ProcessGroup]=None, coordinator_rank: int=0, no_dist: bool=False, planner: Optional[SavePlanner]=None) -> Metadata:
    torch._C._log_api_usage_once('torch.distributed.checkpoint.save_state_dict')
    distW = _DistWrapper(process_group, not no_dist, coordinator_rank)
    if planner is None:
        planner = DefaultSavePlanner()
    assert planner is not None
    global_metatadata = None

    def local_step():
        assert planner is not None
        planner.set_up_planner(state_dict, distW.is_coordinator)
        storage_writer.set_up_storage_writer(distW.is_coordinator)
        local_plan = planner.create_local_plan()
        local_plan = storage_writer.prepare_local_plan(local_plan)
        return local_plan

    def global_step(all_local_plans):
        nonlocal global_metatadata
        assert planner is not None
        all_local_plans, global_metatadata = planner.create_global_plan(all_local_plans)
        all_local_plans = storage_writer.prepare_global_plan(all_local_plans)
        return all_local_plans
    central_plan = distW.reduce_scatter('plan', local_step, global_step)

    def write_data():
        assert planner is not None
        final_local_plan = planner.finish_plan(central_plan)
        all_writes = storage_writer.write_data(final_local_plan, planner)
        all_writes.wait()
        return all_writes.value()

    def finish_checkpoint(all_results):
        assert global_metatadata is not None
        storage_writer.finish(metadata=global_metatadata, results=all_results)
        return global_metatadata
    return distW.all_reduce('write', write_data, finish_checkpoint)