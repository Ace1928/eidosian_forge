import sys
import os
import time
import uuid
import shlex
import threading
import shutil
import subprocess
import logging
import inspect
import ctypes
import runpy
import requests
import psutil
import multiprocess
from dash.testing.errors import (
from dash.testing import wait
@staticmethod
def accessible(url):
    try:
        requests.get(url)
    except requests.exceptions.RequestException:
        return False
    return True