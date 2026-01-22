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
def _generate_machine_readable_data(groups, conf):
    """Create data structure for machine readable sample config

    Returns a dictionary with the top-level keys 'options',
    'deprecated_options', and 'generator_options'.

    'options' contains a dict mapping group names to a list of options in
    that group.  Each option is represented by the result of a call to
    _build_entry.  Only non-deprecated options are included in this list.

    'deprecated_options' contains a dict mapping groups names to a list of
    opts from that group which were deprecated.

    'generator_options' is a dict mapping the options for the sample config
    generator itself to their values.

    :param groups: A dict of groups as returned by _get_groups.
    :param conf: The ConfigOpts object containing the options for the
                 generator tool
    """
    output_data = {'options': {}, 'deprecated_options': {}, 'generator_options': {}}
    for group_name, group_data in groups.items():
        output_group = {'opts': [], 'help': ''}
        output_data['options'][group_name] = output_group
        for namespace in group_data['namespaces']:
            for opt in namespace[1]:
                if group_data['object']:
                    output_group.update(group_data['object']._get_generator_data())
                else:
                    output_group.update({'dynamic_group_owner': '', 'driver_option': '', 'driver_opts': {}})
                entry = _build_entry(opt, group_name, namespace[0], conf)
                output_group['opts'].append(entry)
                for deprecated_opt in copy.deepcopy(entry['deprecated_opts']):
                    group = deprecated_opt.pop('group')
                    deprecated_options = output_data['deprecated_options']
                    deprecated_options.setdefault(group, [])
                    deprecated_opt['replacement_name'] = entry['name']
                    deprecated_opt['replacement_group'] = group_name
                    deprecated_options[group].append(deprecated_opt)
        non_driver_opt_names = [o['name'] for o in output_group['opts'] if not any((o['name'] in output_group['driver_opts'][d] for d in output_group['driver_opts']))]
        output_group['standard_opts'] = non_driver_opt_names
    output_data['generator_options'] = dict(conf)
    return output_data