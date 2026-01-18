from contextlib import contextmanager
import os
import shutil
import sys
import time
import logging
import inspect
import pprint
import subprocess
import textwrap
def toc(self, ok=True):
    if self.counting:
        t = self.timer() - self.t
        if self.msg is not None:
            if ok:
                status = 'ok'
                color = c.ok
            else:
                status = 'error'
                color = c.fail
            self.out.write('%{}s\n'.format(shutil.get_terminal_size()[0] - len(self.msg)) % ('(%{fmt_s} s) [{c}%5s{r}]'.format(fmt_s=self.fmt_s, c=color, r=c.endc) % (t, status)))
            self.out.flush()
        return t
    else:
        raise ValueError('Not counting, did you forget to call ``.tic()`` method?')