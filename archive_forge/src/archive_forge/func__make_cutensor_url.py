import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
def _make_cutensor_url(platform, filename):
    return 'https://developer.download.nvidia.com/compute/cutensor/' + f'redist/libcutensor/{platform}-x86_64/{filename}'