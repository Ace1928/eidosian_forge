import importlib.metadata
import logging
import re
import sys
import yaml
from oslo_config import cfg
from oslo_config import generator
def _validate_defaults(sections, opt_data, conf):
    """Compares the current and sample configuration and reports differences

    :param section: ConfigParser instance
    :param opt_data: machine readable data from the generator instance
    :param conf: ConfigOpts instance
    :returns: boolean wether or not warnings were reported
    """
    warnings = False
    exclusion_regexes = []
    for pattern in conf.exclude_options:
        exclusion_regexes.append(re.compile(pattern))
    for group, opts in opt_data['options'].items():
        if group in conf.exclude_group:
            continue
        if group not in sections:
            logging.warning('Group %s from the sample config is not defined in input-file', group)
            continue
        for opt in opts['opts']:
            if not isinstance(opt['default'], list):
                defaults = [str(opt['default'])]
            else:
                defaults = opt['default']
            opt_names = set([opt['name'], opt.get('dest')])
            if not opt_names.intersection(sections[group]):
                continue
            try:
                value = sections[group][opt['name']]
                keyname = opt['name']
            except KeyError:
                value = sections[group][opt.get('dest')]
                keyname = opt.get('dest')
            if any((rex.fullmatch(keyname) for rex in exclusion_regexes)):
                logging.info('%s/%s Ignoring option because it is part of the excluded patterns. This can be changed with the --exclude-options argument', group, keyname)
                continue
            if len(value) > 1:
                logging.info('%s/%s defined %s times', group, keyname, len(value))
            if not opt['default']:
                logging.warning('%s/%s sample value is empty but input-file has %s', group, keyname, ', '.join(value))
                warnings = True
            elif not frozenset(defaults).intersection(value):
                logging.warning('%s/%s sample value %s is not in %s', group, keyname, defaults, value)
                warnings = True
    return warnings