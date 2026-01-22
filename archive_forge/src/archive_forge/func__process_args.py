import sys, os
import textwrap
def _process_args(self, largs, rargs, values):
    """_process_args(largs : [string],
                         rargs : [string],
                         values : Values)

        Process command-line arguments and populate 'values', consuming
        options and arguments from 'rargs'.  If 'allow_interspersed_args' is
        false, stop at the first non-option argument.  If true, accumulate any
        interspersed non-option arguments in 'largs'.
        """
    while rargs:
        arg = rargs[0]
        if arg == '--':
            del rargs[0]
            return
        elif arg[0:2] == '--':
            self._process_long_opt(rargs, values)
        elif arg[:1] == '-' and len(arg) > 1:
            self._process_short_opts(rargs, values)
        elif self.allow_interspersed_args:
            largs.append(arg)
            del rargs[0]
        else:
            return