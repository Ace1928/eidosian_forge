import pyopencl as cl
import numpy as np
import functools
import os
import logging
from typing import Any, Dict, Tuple
from collections import deque
import pickle
import shutil
import OpenGL.GL as gl

        Compiles a shader from source code using OpenGL.

        Parameters:
            source (str): The shader source code.
            shader_type (str): The type of shader to compile.

        Returns:
            Any: The compiled OpenGL shader.
        