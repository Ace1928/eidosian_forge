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
def _get_neuron_index(self, neurons_df: pd.DataFrame, label: str) -> int:
    """
        Retrieves the index of a neuron from the neurons DataFrame based on its label.
        """
    return int(neurons_df[neurons_df['Label'] == label]['Index'].values[0])