import os
import sys
import msvcrt
import _winapi
from pickle import load
from multiprocessing import process, util
from multiprocessing.context import set_spawning_popen
from multiprocessing.popen_spawn_win32 import Popen as _Popen
from . import reduction, spawn
Run code specified by data received over pipe.