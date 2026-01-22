from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import tensorflow as tf
from google.protobuf import text_format
Converts checkpoint from TF 1.x to TF 2.0 for CannedEstimator.

  Args:
    estimator_type: The type of estimator to be converted. So far, the allowed
      args include 'dnn', 'linear', and 'combined'.
    source_checkpoint: Path to the source checkpoint file to be read in.
    source_graph: Path to the source graph file to be read in.
    target_checkpoint: Path to the target checkpoint to be written out.
  