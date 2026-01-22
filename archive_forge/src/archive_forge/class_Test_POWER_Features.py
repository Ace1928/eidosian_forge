import sys, platform, re, pytest
from numpy.core._multiarray_umath import (
import numpy as np
import subprocess
import pathlib
import os
import re
@pytest.mark.skipif(not is_linux or not is_power, reason='Only for Linux and Power')
class Test_POWER_Features(AbstractTest):
    features = ['VSX', 'VSX2', 'VSX3', 'VSX4']
    features_map = dict(VSX2='ARCH_2_07', VSX3='ARCH_3_00', VSX4='ARCH_3_1')

    def load_flags(self):
        self.load_flags_auxv()