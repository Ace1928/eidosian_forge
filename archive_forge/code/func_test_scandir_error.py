import datetime
import warnings
import random
import string
import tempfile
import os
import contextlib
import json
import urllib.request
import hashlib
import time
import subprocess as sp
import multiprocessing as mp
import platform
import pickle
import zipfile
import re
import av
import pytest
from tensorflow.io import gfile
import imageio
import numpy as np
import blobfile as bf
from blobfile import _ops as ops, _azure as azure, _common as common
def test_scandir_error():
    for error, path in [(None, AZURE_VALID_CONTAINER), (FileNotFoundError, AZURE_INVALID_CONTAINER), (FileNotFoundError, AZURE_INVALID_CONTAINER_NO_ACCOUNT), (bf.Error, f'https://{AS_EXTERNAL_ACCOUNT}.blob.core.windows.net/private')]:
        ctx = contextlib.nullcontext()
        if error is not None:
            ctx = pytest.raises(error)
        with ctx:
            print(path)
            list(bf.scandir(path))