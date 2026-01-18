from __future__ import annotations
import time
import os, os.path, sys, glob, argparse, resource, multiprocessing
import pandas as pd
import dask.dataframe as dd
import numpy as np
import datashader as ds
import feather
import fastparquet as fp
from datashader.utils import export_image
from datashader import transfer_functions as tf
from dask import distributed
def write_feather_dask(filepath, df):
    return feather.write_dataframe(df.compute(), filepath)