from __future__ import (absolute_import, division, print_function)
import os
import copy
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
def validate_ldap_sync_config(config):
    url = config.get('url')
    if not url:
        return 'url should be non empty attribute.'
    bind_dn = config.get('bindDN', '')
    bind_password = config.get('bindPassword', '')
    if (len(bind_dn) == 0) != (len(bind_password) == 0):
        return 'bindDN and bindPassword must both be specified, or both be empty.'
    insecure = boolean(config.get('insecure'))
    ca_file = config.get('ca')
    if insecure:
        if url.startswith('ldaps://'):
            return 'Cannot use ldaps scheme with insecure=true.'
        if ca_file:
            return 'Cannot specify a ca with insecure=true.'
    elif ca_file and (not os.path.isfile(ca_file)):
        return 'could not read ca file: {0}.'.format(ca_file)
    nameMapping = config.get('groupUIDNameMapping', {})
    for k, v in iteritems(nameMapping):
        if len(k) == 0 or len(v) == 0:
            return 'groupUIDNameMapping has empty key or value'
    schemas = []
    schema_list = ('rfc2307', 'activeDirectory', 'augmentedActiveDirectory')
    for schema in schema_list:
        if schema in config:
            schemas.append(schema)
    if len(schemas) == 0:
        return 'No schema-specific config was provided, should be one of %s' % ', '.join(schema_list)
    if len(schemas) > 1:
        return 'Exactly one schema-specific config is required; found (%d) %s' % (len(schemas), ','.join(schemas))
    if schemas[0] == 'rfc2307':
        return validate_RFC2307(config.get('rfc2307'))
    elif schemas[0] == 'activeDirectory':
        return validate_ActiveDirectory(config.get('activeDirectory'))
    elif schemas[0] == 'augmentedActiveDirectory':
        return validate_AugmentedActiveDirectory(config.get('augmentedActiveDirectory'))