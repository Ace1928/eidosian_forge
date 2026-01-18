import os
import sys
def to_argv(self, lst, setup):
    v = setup.get(self.arg_name)
    if v:
        lst.append(self.arg_v_rep)