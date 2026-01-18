import argparse
import numpy as np
from ray.rllib.policy.sample_batch import convert_ma_batch_to_sample_batch
from ray.rllib.algorithms import cql as cql
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.execution.rollout_ops import (
Example on how to use CQL to learn from an offline json file.

Important node: Make sure that your offline data file contains only
a single timestep per line to mimic the way SAC pulls samples from
the buffer.

Generate the offline json file by running an SAC algo until it reaches expert
level on your command line. For example:
$ cd ray
$ rllib train -f rllib/tuned_examples/sac/pendulum-sac.yaml --no-ray-ui

Also make sure that in the above SAC yaml file (pendulum-sac.yaml),
you specify an additional "output" key with any path on your local
file system. In that path, the offline json files will be written to.

Use the generated file(s) as "input" in the CQL config below
(`config["input"] = [list of your json files]`), then run this script.
