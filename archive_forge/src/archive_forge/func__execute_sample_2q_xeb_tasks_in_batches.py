import concurrent
import os
import time
import uuid
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass
from typing import (
import numpy as np
import pandas as pd
import tqdm
from cirq import ops, devices, value, protocols
from cirq.circuits import Circuit, Moment
from cirq.experiments.random_quantum_circuit_generation import CircuitLibraryCombination
def _execute_sample_2q_xeb_tasks_in_batches(tasks: List[_Sample2qXEBTask], sampler: 'cirq.Sampler', combinations_by_layer: List[CircuitLibraryCombination], repetitions: int, batch_size: int, progress_bar: Callable[..., ContextManager], dataset_directory: Optional[str]=None) -> List[Dict[str, Any]]:
    """Helper function used in `sample_2q_xeb_circuits` to batch and execute sampling tasks."""
    n_tasks = len(tasks)
    batched_tasks = [tasks[i:i + batch_size] for i in range(0, n_tasks, batch_size)]
    run_batch = _SampleInBatches(sampler=sampler, repetitions=repetitions, combinations_by_layer=combinations_by_layer)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_batch, task_batch) for task_batch in batched_tasks]
        records = []
        with progress_bar(total=len(batched_tasks) * batch_size) as progress:
            for future in concurrent.futures.as_completed(futures):
                new_records = future.result()
                if dataset_directory is not None:
                    os.makedirs(f'{dataset_directory}', exist_ok=True)
                    protocols.to_json(new_records, f'{dataset_directory}/xeb.{uuid.uuid4()}.json')
                records.extend(new_records)
                progress.update(batch_size)
    return records