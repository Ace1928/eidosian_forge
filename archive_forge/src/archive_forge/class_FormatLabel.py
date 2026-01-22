from __future__ import division
import datetime
import math
class FormatLabel(Timer):
    """Displays a formatted label."""
    mapping = {'elapsed': ('seconds_elapsed', Timer.format_time), 'finished': ('finished', None), 'last_update': ('last_update_time', None), 'max': ('maxval', None), 'seconds': ('seconds_elapsed', None), 'start': ('start_time', None), 'value': ('currval', None)}
    __slots__ = ('format_string',)

    def __init__(self, format):
        self.format_string = format

    def update(self, pbar):
        context = {}
        for name, (key, transform) in self.mapping.items():
            try:
                value = getattr(pbar, key)
                if transform is None:
                    context[name] = value
                else:
                    context[name] = transform(value)
            except:
                pass
        return self.format_string % context