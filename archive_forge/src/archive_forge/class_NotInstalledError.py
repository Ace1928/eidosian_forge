import queue
import re
import subprocess
import sys
import threading
import time
from io import DEFAULT_BUFFER_SIZE
from .exceptions import DecodeError
from .base import AudioFile
class NotInstalledError(FFmpegError):
    """Could not find the ffmpeg binary."""