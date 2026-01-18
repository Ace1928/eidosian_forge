import sys, platform, re, pytest
from numpy.core._multiarray_umath import (
import numpy as np
import subprocess
import pathlib
import os
import re

        Test that a RuntimeError is thrown if an impossible feature-enabling
        request is made. This includes enabling a feature not supported by the
        machine, or disabling a baseline optimization.
        