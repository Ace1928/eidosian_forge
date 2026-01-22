import argparse
import json
import os
import re
import shutil
import yaml

This script automates cleaning up a benchmark/experiment run of some algo
against some config (with possibly more than one tune trial,
e.g. torch=grid_search([True, False])).

Run `python cleanup_experiment.py --help` for more information.

Use on an input directory with trial contents e.g.:
..
IMPALA_BreakoutNoFrameskip-v4_0_use_pytorch=False_2020-05-11_10-17-54topr3h9k
IMPALA_BreakoutNoFrameskip-v4_0_use_pytorch=False_2020-05-11_13-59-35dqaetxnf
IMPALA_BreakoutNoFrameskip-v4_0_use_pytorch=False_2020-05-11_17-21-28tbhedw72
IMPALA_BreakoutNoFrameskip-v4_2_use_pytorch=True_2020-05-11_10-17-54lv20cgn_
IMPALA_BreakoutNoFrameskip-v4_2_use_pytorch=True_2020-05-11_13-59-35kwzhax_y
IMPALA_BreakoutNoFrameskip-v4_2_use_pytorch=True_2020-05-11_17-21-28a5j0s7za

Then run:
>> python cleanup_experiment.py --experiment-dir [parent dir w/ trial sub-dirs]
>>   --output-dir [your out dir] --results-filter dumb_col_2,superfluous_col3
>>   --results-max-size [max results file size in kb before(!) zipping]

The script will create one output sub-dir for each trial and only copy
the configuration and the csv results (filtered and every nth row removed
based on the given args).
