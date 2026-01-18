import re
import datetime
import numpy as np
import csv
import ctypes
def test_weka(filename):
    data, meta = loadarff(filename)
    print(len(data.dtype))
    print(data.size)
    for i in meta:
        print_attribute(i, meta[i], data[i])