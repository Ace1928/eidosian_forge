from __future__ import (absolute_import, division, print_function)
import os
import copy
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
class OpenshiftLDAPQueryOnAttribute(object):

    def __init__(self, qry, attribute):
        self.qry = copy.deepcopy(qry)
        self.query_attribute = attribute

    @staticmethod
    def escape_filter(buffer):
        """
        escapes from the provided LDAP filter string the special
        characters in the set '(', ')', '*', \\ and those out of the range 0 < c < 0x80, as defined in RFC4515.
        """
        output = []
        hex_string = '0123456789abcdef'
        for c in buffer:
            if ord(c) > 127 or c in ('(', ')', '\\', '*') or c == 0:
                first = ord(c) >> 4
                second = ord(c) & 15
                output += ['\\', hex_string[first], hex_string[second]]
            else:
                output.append(c)
        return ''.join(output)

    def build_request(self, ldapuid, attributes):
        params = copy.deepcopy(self.qry)
        if self.query_attribute.lower() == 'dn':
            if ldapuid:
                if not openshift_equal_dn(ldapuid, params['base']) and (not openshift_ancestorof_dn(params['base'], ldapuid)):
                    return (None, LDAP_SEARCH_OUT_OF_SCOPE_ERROR)
                params['base'] = ldapuid
            params['scope'] = ldap.SCOPE_BASE
            params['filterstr'] = '(objectClass=*)'
            params['attrlist'] = attributes
        else:
            specificFilter = '%s=%s' % (self.escape_filter(self.query_attribute), self.escape_filter(ldapuid))
            qry_filter = params.get('filterstr', None)
            if qry_filter:
                params['filterstr'] = '(&%s(%s))' % (qry_filter, specificFilter)
            params['attrlist'] = attributes
        return (params, None)

    def ldap_search(self, connection, ldapuid, required_attributes, unique_entry=True):
        query, error = self.build_request(ldapuid, required_attributes)
        if error:
            return (None, error)
        derefAlias = query.pop('derefAlias', None)
        if derefAlias:
            ldap.set_option(ldap.OPT_DEREF, derefAlias)
        try:
            result = connection.search_ext_s(**query)
            if not result or len(result) == 0:
                return (None, "Entry not found for base='{0}' and filter='{1}'".format(query['base'], query['filterstr']))
            if unique_entry:
                if len(result) > 1:
                    return (None, 'Multiple Entries found matching search criteria: %s (%s)' % (query, result))
                result = result[0]
            return (result, None)
        except ldap.NO_SUCH_OBJECT:
            return (None, "Entry not found for base='{0}' and filter='{1}'".format(query['base'], query['filterstr']))
        except Exception as err:
            return (None, 'Request %s failed due to: %s' % (query, err))