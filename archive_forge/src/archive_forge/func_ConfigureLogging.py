import logging
import sys
from typing import Optional, TextIO
from absl import flags
from absl import logging as absl_logging
from googleapiclient import model
def ConfigureLogging(apilog: Optional[str]=None):
    try:
        ConfigurePythonLogger(apilog)
    except IOError as e:
        if e.errno == 2:
            print('Could not configure logging. %s: %s' % (e.strerror, e.filename))
            sys.exit(1)
        raise e