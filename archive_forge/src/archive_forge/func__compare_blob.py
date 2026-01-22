import argparse
import logging
import os
import warnings
import numpy as np
import cv2
import mxnet as mx
def _compare_blob(caf_blob, mx_blob, caf_name, mx_name, blob_type, note):
    diff = np.abs(mx_blob - caf_blob)
    diff_mean = diff.mean()
    diff_max = diff.max()
    logging.info(log_format.format(caf_name, mx_name, blob_type, '%4.5f' % diff_mean, '%4.5f' % diff_max, note))
    assert diff_mean < mean_diff_allowed
    assert diff_max < max_diff_allowed