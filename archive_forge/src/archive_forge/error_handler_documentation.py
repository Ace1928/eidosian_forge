import faulthandler
import json
import logging
import os
import time
import traceback
import warnings
from typing import Any, Dict, Optional
Dump parent error file from child process's root cause error and error code.