import asyncio
import contextlib
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from queue import Empty, Queue
from threading import Thread
import numpy as np
import pytest
import requests
from packaging.version import Version
import panel as pn
from panel.io.server import serve
from panel.io.state import state
from panel.pane.alert import Alert
from panel.pane.markup import Markdown
from panel.widgets.button import _ButtonBase
def timed_out():
    elapsed = time.time() - start
    elapsed_ms = elapsed * 1000
    return elapsed_ms > timeout