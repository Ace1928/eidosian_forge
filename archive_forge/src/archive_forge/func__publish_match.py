import re
from yaql.language import specs
from yaql.language import yaqltypes
def _publish_match(context, match):
    rec = {'value': match.group(), 'start': match.start(0), 'end': match.end(0)}
    context['$1'] = rec
    for i, t in enumerate(match.groups(), 1):
        rec = {'value': t, 'start': match.start(i), 'end': match.end(i)}
        context['$' + str(i + 1)] = rec
    for key, value in match.groupdict().values():
        rec = {'value': value, 'start': match.start(value), 'end': match.end(value)}
        context['$' + key] = rec