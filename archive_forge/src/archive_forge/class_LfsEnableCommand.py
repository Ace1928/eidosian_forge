import json
import os
import subprocess
import sys
import warnings
from argparse import ArgumentParser
from contextlib import AbstractContextManager
from typing import Dict, List, Optional
import requests
from ..utils import logging
from . import BaseTransformersCLICommand
class LfsEnableCommand:

    def __init__(self, args):
        self.args = args

    def run(self):
        warnings.warn('Managing repositories through transformers-cli is deprecated. Please use `huggingface-cli` instead.')
        local_path = os.path.abspath(self.args.path)
        if not os.path.isdir(local_path):
            print('This does not look like a valid git repo.')
            exit(1)
        subprocess.run('git config lfs.customtransfer.multipart.path transformers-cli'.split(), check=True, cwd=local_path)
        subprocess.run(f'git config lfs.customtransfer.multipart.args {LFS_MULTIPART_UPLOAD_COMMAND}'.split(), check=True, cwd=local_path)
        print('Local repo set up for largefiles')