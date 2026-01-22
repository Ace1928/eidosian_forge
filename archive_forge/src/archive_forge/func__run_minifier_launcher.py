import dataclasses
import io
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Optional
from unittest.mock import patch
import torch
import torch._dynamo
import torch._dynamo.test_case
from torch.utils._traceback import report_compile_source_on_error
import torch
import torch._dynamo
def _run_minifier_launcher(self, repro_dir, isolate, *, minifier_args=()):
    self.assertIsNotNone(repro_dir)
    launch_file = os.path.join(repro_dir, 'minifier_launcher.py')
    with open(launch_file) as f:
        launch_code = f.read()
    self.assertTrue(os.path.exists(launch_file))
    args = ['python3', launch_file, 'minify', *minifier_args]
    if not isolate:
        args.append('--no-isolate')
    launch_proc = self._maybe_subprocess_run(args, isolate=isolate, cwd=repro_dir)
    print('minifier stdout:', launch_proc.stdout.decode('utf-8'))
    stderr = launch_proc.stderr.decode('utf-8')
    print('minifier stderr:', stderr)
    self.assertNotIn('Input graph did not fail the tester', stderr)
    return (launch_proc, launch_code)