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
def _cleanup_opts(read_opts):
    """Cleanup duplicate options in namespace groups

    Return a structure which removes duplicate options from a namespace group.
    NOTE:(rbradfor) This does not remove duplicated options from repeating
    groups in different namespaces:

    :param read_opts: a list (namespace, [(group, [opt_1, opt_2])]) tuples
    :returns: a list of (namespace, [(group, [opt_1, opt_2])]) tuples
    """
    clean = collections.OrderedDict()
    for namespace, listing in read_opts:
        if namespace not in clean:
            clean[namespace] = collections.OrderedDict()
        for group, opts in listing:
            if group:
                group_name = getattr(group, 'name', str(group))
                if group_name.upper() in UPPER_CASE_GROUP_NAMES:
                    normalized_gn = group_name.upper()
                else:
                    normalized_gn = group_name.lower()
                if normalized_gn != group_name:
                    LOG.warning('normalizing group name %r to %r', group_name, normalized_gn)
                    if hasattr(group, 'name'):
                        group.name = normalized_gn
                    else:
                        group = normalized_gn
            if group not in clean[namespace]:
                clean[namespace][group] = collections.OrderedDict()
            for opt in opts:
                clean[namespace][group][opt.dest] = opt
    cleaned_opts = [(namespace, [(g, list(clean[namespace][g].values())) for g in clean[namespace]]) for namespace in clean]
    return cleaned_opts