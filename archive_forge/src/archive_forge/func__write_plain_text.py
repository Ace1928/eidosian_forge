import itertools
import logging
import os
import queue
import re
import signal
import struct
import sys
import threading
import time
from collections import defaultdict
import wandb
def _write_plain_text(self, plain_text):
    self.buffer[self.cursor.y].update([(self.cursor.x + i, self.cursor.char.copy(data=c)) for i, c in enumerate(plain_text)])
    self.cursor.x += len(plain_text)