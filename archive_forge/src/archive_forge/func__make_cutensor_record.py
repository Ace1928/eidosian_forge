import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
def _make_cutensor_record(cuda_version):
    return __make_cutensor_record(cuda_version, '1.6.2', 'libcutensor-linux-x86_64-1.6.2.3-archive.tar.xz', 'libcutensor-windows-x86_64-1.6.2.3-archive.zip')