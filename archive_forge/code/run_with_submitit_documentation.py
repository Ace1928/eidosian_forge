import argparse
import os
import uuid
from pathlib import Path
import submitit
from xformers.benchmarks.LRA.run_tasks import benchmark, get_arg_parser

A script to run multinode training with submitit.
Almost copy-paste from https://github.com/facebookresearch/deit/blob/main/run_with_submitit.py
