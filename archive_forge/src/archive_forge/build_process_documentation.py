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
dash-renderer's path is binding with the dash folder hierarchy.