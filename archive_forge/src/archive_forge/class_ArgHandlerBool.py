import os
import sys
class ArgHandlerBool:
    """
    If a given flag is received, mark it as 'True' in setup.
    """

    def __init__(self, arg_name, default_val=False):
        self.arg_name = arg_name
        self.arg_v_rep = '--%s' % (arg_name,)
        self.default_val = default_val

    def to_argv(self, lst, setup):
        v = setup.get(self.arg_name)
        if v:
            lst.append(self.arg_v_rep)

    def handle_argv(self, argv, i, setup):
        assert argv[i] == self.arg_v_rep
        del argv[i]
        setup[self.arg_name] = True