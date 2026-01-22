import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
def _make_cudnn_url(platform, filename):
    return 'https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/' + f'{platform}/{filename}'