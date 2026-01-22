import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
def _make_nccl_record(cuda_version, full_version, public_version, filename_linux):
    return {'cuda': cuda_version, 'nccl': full_version, 'assets': {'Linux': {'url': _make_nccl_url(public_version, filename_linux), 'filenames': ['libnccl.so.{}'.format(full_version)]}}}