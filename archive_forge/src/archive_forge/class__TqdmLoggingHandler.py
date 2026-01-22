import logging
import sys
from contextlib import contextmanager
from ..std import tqdm as std_tqdm
class _TqdmLoggingHandler(logging.StreamHandler):

    def __init__(self, tqdm_class=std_tqdm):
        super(_TqdmLoggingHandler, self).__init__()
        self.tqdm_class = tqdm_class

    def emit(self, record):
        try:
            msg = self.format(record)
            self.tqdm_class.write(msg, file=self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)