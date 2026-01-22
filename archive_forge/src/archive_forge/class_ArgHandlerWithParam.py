import os
import sys
class ArgHandlerWithParam:
    """
    Handler for some arguments which needs a value
    """

    def __init__(self, arg_name, convert_val=None, default_val=None):
        self.arg_name = arg_name
        self.arg_v_rep = '--%s' % (arg_name,)
        self.convert_val = convert_val
        self.default_val = default_val

    def to_argv(self, lst, setup):
        v = setup.get(self.arg_name)
        if v is not None and v != self.default_val:
            lst.append(self.arg_v_rep)
            lst.append('%s' % (v,))

    def handle_argv(self, argv, i, setup):
        assert argv[i] == self.arg_v_rep
        del argv[i]
        val = argv[i]
        if self.convert_val:
            val = self.convert_val(val)
        setup[self.arg_name] = val
        del argv[i]