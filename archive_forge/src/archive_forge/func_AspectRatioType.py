from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AspectRatioType(value):
    """A type function to be used to parse aspect ratios."""
    try:
        return float(value)
    except ValueError:
        parts = value.split(':')
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1])
            except ValueError:
                pass
        raise arg_parsers.ArgumentTypeError('Each aspect ratio must either be specified as a decimal (ex. 1.333) or as a ratio of width to height (ex 4:3)')