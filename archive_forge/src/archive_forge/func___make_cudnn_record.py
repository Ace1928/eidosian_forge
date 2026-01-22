import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
def __make_cudnn_record(cuda_version, public_version, filename_linux, filename_windows):
    major_version = public_version.split('.')[0]
    suffix_list = ['', '_ops_infer', '_ops_train', '_cnn_infer', '_cnn_train', '_adv_infer', '_adv_train']
    return {'cuda': cuda_version, 'cudnn': public_version, 'assets': {'Linux': {'url': _make_cudnn_url('linux-x86_64', filename_linux), 'filenames': [f'libcudnn{suffix}.so.{public_version}' for suffix in suffix_list]}, 'Windows': {'url': _make_cudnn_url('windows-x86_64', filename_windows), 'filenames': [f'cudnn{suffix}64_{major_version}.dll' for suffix in suffix_list]}}}