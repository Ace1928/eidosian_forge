import argparse
import itertools as it
import multiprocessing
import multiprocessing.dummy
import os
import pickle
import queue
import subprocess
import tempfile
import textwrap
import numpy as np
import torch
from torch.utils.benchmark.op_fuzzers import unary
from torch.utils.benchmark import Timer, Measurement
from typing import Dict, Tuple, List
def _main(args):
    pools, map_iters, finished_counts = ({}, {}, {})
    pr = args.pr
    envs = (_REF_ENV_TEMPLATE.format(pr=pr), _PR_ENV_TEMPLATE.format(pr=pr))
    if _DEVICES_TO_TEST[args.pr][_GPU]:
        finished_counts[_GPU] = 0
        for i in range(args.num_gpus):
            _AVAILABLE_GPUS.put(i)
        pools[_GPU] = multiprocessing.dummy.Pool(args.num_gpus)
        trials = [(seed, envs, pr, True, finished_counts, args.test_variance) for seed in range(_NUM_LOOPS[_GPU])] * _REPLICATES[_GPU]
        map_iters[_GPU] = pools[_GPU].imap(map_fn, trials)
    if _DEVICES_TO_TEST[args.pr][_CPU]:
        finished_counts[_CPU] = 0
        cpu_workers = int(multiprocessing.cpu_count() / 3)
        pools[_CPU] = multiprocessing.dummy.Pool(cpu_workers)
        trials = [(seed, envs, pr, False, finished_counts, args.test_variance) for seed in range(_NUM_LOOPS[_CPU])] * _REPLICATES[_CPU]
        map_iters[_CPU] = pools[_CPU].imap(map_fn, trials)
    results = []
    for map_iter in map_iters.values():
        for r in map_iter:
            results.append(r)
            progress = [f'{k}: {v} / {_NUM_LOOPS[k] * _REPLICATES[k]}' for k, v in finished_counts.items()]
            print(f'\r{(' ' * 10).join(progress)}', end='')
    print()
    for pool in pools.values():
        pool.close()
    process_results(results, args.test_variance)