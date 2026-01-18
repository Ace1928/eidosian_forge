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
def test_pickle_config():
    ctx = ops.create_context()
    c = ctx._conf
    pickle.dumps(c)
    c.get_http_pool()
    c2 = pickle.loads(pickle.dumps(c))
    c2.get_http_pool()