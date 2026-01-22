from tqdm import tqdm, tqdm_notebook
from collections import OrderedDict
import time
class MuteProgressBarLogger(ProgressBarLogger):

    def bar_is_ignored(self, bar):
        return True