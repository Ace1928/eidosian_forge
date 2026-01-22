import sys
def _processoptions(args):
    for arg in args:
        try:
            _setoption(arg)
        except _OptionError as msg:
            print('Invalid -W option ignored:', msg, file=sys.stderr)