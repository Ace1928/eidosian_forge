import queue
import re
import subprocess
import sys
import threading
import time
from io import DEFAULT_BUFFER_SIZE
from .exceptions import DecodeError
from .base import AudioFile
Close the ffmpeg process used to perform the decoding.