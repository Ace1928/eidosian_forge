import collections
from .utils import ExplicitEnum, is_torch_available, logging
def trace_frames(self):
    print('\n'.join(self.frames))
    self.frames = []