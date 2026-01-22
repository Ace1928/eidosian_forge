import os
import sys
import json
import string
import shutil
import logging
import coloredlogs
import fire
import requests
from .._utils import run_command_with_process, compute_md5, job
def _parse_package(self, path):
    with open(path, 'r', encoding='utf-8') as fp:
        package = json.load(fp)
        self.version = package['version']
        self.name = package['name']
        self.deps_folder = self._concat(self.main, os.pardir, 'deps')
        self.deps = package['dependencies']