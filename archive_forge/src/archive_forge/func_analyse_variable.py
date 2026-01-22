import collections
from .utils import ExplicitEnum, is_torch_available, logging
def analyse_variable(self, var, ctx):
    if torch.is_tensor(var):
        self.expand_frame(get_abs_min_max(var, ctx))
        if detect_overflow(var, ctx):
            self.detected_overflow = True
    elif var is None:
        self.expand_frame(f'{'None':>17} {ctx}')
    else:
        self.expand_frame(f'{'not a tensor':>17} {ctx}')