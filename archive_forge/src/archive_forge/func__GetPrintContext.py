from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Optional
from utils import bq_error
def _GetPrintContext(self) -> str:
    return '%r' % (self.job_ref,)