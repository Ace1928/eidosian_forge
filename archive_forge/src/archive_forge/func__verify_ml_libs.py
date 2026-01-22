from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import json
import sys
def _verify_ml_libs(framework):
    """Verifies the appropriate ML libs are installed per framework."""
    if framework == 'tensorflow' and (not _verify_tensorflow('1.0.0')):
        sys.exit(-1)
    elif framework == 'scikit_learn' and (not _verify_scikit_learn('0.18.1')):
        sys.exit(-1)
    elif framework == 'xgboost' and (not _verify_xgboost('0.6a2')):
        sys.exit(-1)