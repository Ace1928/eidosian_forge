import argparse
from osc_lib.i18n import _
class RangeAction(argparse.Action):
    """A custom action to parse a single value or a range of values

    Parses single integer values or a range of integer values delimited
    by a colon and returns a tuple of integers:
    '4' sets ``dest`` to (4, 4)
    '6:9' sets ``dest`` to (6, 9)
    """

    def __call__(self, parser, namespace, values, option_string=None):
        range = values.split(':')
        if len(range) == 0:
            setattr(namespace, self.dest, (0, 0))
        elif len(range) == 1:
            setattr(namespace, self.dest, (int(range[0]), int(range[0])))
        elif len(range) == 2:
            if int(range[0]) <= int(range[1]):
                setattr(namespace, self.dest, (int(range[0]), int(range[1])))
            else:
                msg = _('Invalid range, %(min)s is not less than %(max)s')
                raise argparse.ArgumentError(self, msg % {'min': range[0], 'max': range[1]})
        else:
            msg = _('Invalid range, too many values')
            raise argparse.ArgumentError(self, msg)