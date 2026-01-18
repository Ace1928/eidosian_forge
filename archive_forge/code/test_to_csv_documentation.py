import io
import os
import sys
from zipfile import ZipFile
from _csv import Error
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm

        Binary file objects should honor a specified encoding.

        GH 23854 and GH 13068 with binary handles
        