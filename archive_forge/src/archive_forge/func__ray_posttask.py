import contextlib
from collections import namedtuple, defaultdict
from datetime import datetime
from dask.callbacks import Callback
def _ray_posttask(self, key, result, pre_state):
    self.pb.finish.remote(key, datetime.now())