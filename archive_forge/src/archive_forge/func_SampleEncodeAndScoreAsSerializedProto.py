from sys import version_info as _swig_python_version_info
import re
import csv
import sys
import os
from io import StringIO
from io import BytesIO
from ._version import __version__
def SampleEncodeAndScoreAsSerializedProto(self, input, num_samples=None, alpha=None, **kwargs):
    return self.SampleEncodeAndScore(input=input, num_samples=num_samples, alpha=alpha, out_type='serialized_proto', **kwargs)