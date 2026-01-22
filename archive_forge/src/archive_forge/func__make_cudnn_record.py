import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
def _make_cudnn_record(cuda_version):
    cuda_major = int(cuda_version.split('.')[0])
    assert cuda_major in (11, 12)
    return __make_cudnn_record(cuda_version, '8.8.1', f'cudnn-linux-x86_64-8.8.1.3_cuda{cuda_major}-archive.tar.xz', f'cudnn-windows-x86_64-8.8.1.3_cuda{cuda_major}-archive.zip')