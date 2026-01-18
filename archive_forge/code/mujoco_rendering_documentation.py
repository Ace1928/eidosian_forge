import collections
import os
import time
from threading import Lock
import glfw
import imageio
import mujoco
import numpy as np
Override close in your rendering subclass to perform any necessary cleanup
        after env.close() is called.
        