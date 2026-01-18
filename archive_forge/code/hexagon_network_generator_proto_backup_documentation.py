import logging
import math
import sys
from typing import List, Tuple, Dict, Optional, Union, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
import pandas as pd
from datetime import datetime

        Retrieves the index of a neuron from the neurons DataFrame based on its label.
        