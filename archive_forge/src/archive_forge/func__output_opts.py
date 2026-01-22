import collections
import copy
import importlib.metadata
import json
import logging
import operator
import sys
import yaml
from oslo_config import cfg
from oslo_i18n import _message
import stevedore.named  # noqa
def _output_opts(f, group, group_data):
    f.format_group(group_data['object'] or group)
    for namespace, opts in sorted(group_data['namespaces'], key=operator.itemgetter(0)):
        f.write('\n#\n# From %s\n#\n' % namespace)
        for opt in sorted(opts, key=operator.attrgetter('advanced')):
            try:
                if f.minimal and (not opt.required):
                    pass
                else:
                    f.write('\n')
                    f.format(opt, group)
            except Exception as err:
                f.write('# Warning: Failed to format sample for %s\n' % (opt.dest,))
                f.write('# %s\n' % (err,))