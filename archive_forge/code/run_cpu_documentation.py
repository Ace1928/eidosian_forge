import glob
import logging
import os
import platform
import re
import subprocess
import sys
from argparse import ArgumentParser, RawTextHelpFormatter, REMAINDER
from os.path import expanduser
from typing import Dict, List
from torch.distributed.elastic.multiprocessing import start_processes, Std

        Set multi-thread configuration and enable Intel openMP and TCMalloc/JeMalloc.

        By default, GNU openMP and PTMalloc are used in PyTorch. but Intel openMP and TCMalloc/JeMalloc are better alternatives
        to get performance benefit.
        