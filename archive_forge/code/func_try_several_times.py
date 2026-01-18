import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def try_several_times(f, t=3, s=1):
    e = RuntimeError()
    for _ in range(t):
        try:
            r = f()
        except RuntimeError as e:
            time.sleep(s)
        else:
            return r
    raise e