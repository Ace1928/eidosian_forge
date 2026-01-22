import logging
import re
from enum import Enum
from string import Formatter
from typing import NamedTuple
from botocore import xform_name
from botocore.compat import IPV4_RE, quote, urlparse
from botocore.exceptions import EndpointResolutionError
from botocore.utils import (
class RuleSetStandardLibrary:
    """Rule actions to be performed by the EndpointProvider."""

    def __init__(self, partitions_data):
        self.partitions_data = partitions_data

    def is_func(self, argument):
        """Determine if an object is a function object.

        :type argument: Any
        :rtype: bool
        """
        return isinstance(argument, dict) and 'fn' in argument

    def is_ref(self, argument):
        """Determine if an object is a reference object.

        :type argument: Any
        :rtype: bool
        """
        return isinstance(argument, dict) and 'ref' in argument

    def is_template(self, argument):
        """Determine if an object contains a template string.

        :type argument: Any
        :rtpe: bool
        """
        return isinstance(argument, str) and TEMPLATE_STRING_RE.search(argument) is not None

    def resolve_template_string(self, value, scope_vars):
        """Resolve and inject values into a template string.

        :type value: str
        :type scope_vars: dict
        :rtype: str
        """
        result = ''
        for literal, reference, _, _ in STRING_FORMATTER.parse(value):
            if reference is not None:
                template_value = scope_vars
                template_params = reference.split('#')
                for param in template_params:
                    template_value = template_value[param]
                result += f'{literal}{template_value}'
            else:
                result += literal
        return result

    def resolve_value(self, value, scope_vars):
        """Return evaluated value based on type.

        :type value: Any
        :type scope_vars: dict
        :rtype: Any
        """
        if self.is_func(value):
            return self.call_function(value, scope_vars)
        elif self.is_ref(value):
            return scope_vars.get(value['ref'])
        elif self.is_template(value):
            return self.resolve_template_string(value, scope_vars)
        return value

    def convert_func_name(self, value):
        """Normalize function names.

        :type value: str
        :rtype: str
        """
        normalized_name = f'{xform_name(value)}'
        if normalized_name == 'not':
            normalized_name = f'_{normalized_name}'
        return normalized_name.replace('.', '_')

    def call_function(self, func_signature, scope_vars):
        """Call the function with the resolved arguments and assign to `scope_vars`
        when applicable.

        :type func_signature: dict
        :type scope_vars: dict
        :rtype: Any
        """
        func_args = [self.resolve_value(arg, scope_vars) for arg in func_signature['argv']]
        func_name = self.convert_func_name(func_signature['fn'])
        func = getattr(self, func_name)
        result = func(*func_args)
        if 'assign' in func_signature:
            assign = func_signature['assign']
            if assign in scope_vars:
                raise EndpointResolutionError(msg=f'Assignment {assign} already exists in scoped variables and cannot be overwritten')
            scope_vars[assign] = result
        return result

    def is_set(self, value):
        """Evaluates whether a value is set.

        :type value: Any
        :rytpe: bool
        """
        return value is not None

    def get_attr(self, value, path):
        """Find an attribute within a value given a path string. The path can contain
        the name of the attribute and an index in brackets. A period separating attribute
        names indicates the one to the right is nested. The index will always occur at
        the end of the path.

        :type value: dict or list
        :type path: str
        :rtype: Any
        """
        for part in path.split('.'):
            match = GET_ATTR_RE.search(part)
            if match is not None:
                name, index = match.groups()
                index = int(index)
                value = value.get(name)
                if value is None or index >= len(value):
                    return None
                return value[index]
            else:
                value = value[part]
        return value

    def format_partition_output(self, partition):
        output = partition['outputs']
        output['name'] = partition['id']
        return output

    def is_partition_match(self, region, partition):
        matches_regex = re.match(partition['regionRegex'], region) is not None
        return region in partition['regions'] or matches_regex

    def aws_partition(self, value):
        """Match a region string to an AWS partition.

        :type value: str
        :rtype: dict
        """
        partitions = self.partitions_data['partitions']
        if value is not None:
            for partition in partitions:
                if self.is_partition_match(value, partition):
                    return self.format_partition_output(partition)
        aws_partition = partitions[0]
        return self.format_partition_output(aws_partition)

    def aws_parse_arn(self, value):
        """Parse and validate string for ARN components.

        :type value: str
        :rtype: dict
        """
        if value is None or not value.startswith('arn:'):
            return None
        try:
            arn_dict = ARN_PARSER.parse_arn(value)
        except InvalidArnException:
            return None
        if not all((arn_dict['partition'], arn_dict['service'], arn_dict['resource'])):
            return None
        arn_dict['accountId'] = arn_dict.pop('account')
        resource = arn_dict.pop('resource')
        arn_dict['resourceId'] = resource.replace(':', '/').split('/')
        return arn_dict

    def is_valid_host_label(self, value, allow_subdomains):
        """Evaluates whether a value is a valid host label per
        RFC 1123. If allow_subdomains is True, split on `.` and validate
        each component separately.

        :type value: str
        :type allow_subdomains: bool
        :rtype: bool
        """
        if value is None or (allow_subdomains is False and value.count('.') > 0):
            return False
        if allow_subdomains is True:
            return all((self.is_valid_host_label(label, False) for label in value.split('.')))
        return VALID_HOST_LABEL_RE.match(value) is not None

    def string_equals(self, value1, value2):
        """Evaluates two string values for equality.

        :type value1: str
        :type value2: str
        :rtype: bool
        """
        if not all((isinstance(val, str) for val in (value1, value2))):
            msg = f'Both values must be strings, not {type(value1)} and {type(value2)}.'
            raise EndpointResolutionError(msg=msg)
        return value1 == value2

    def uri_encode(self, value):
        """Perform percent-encoding on an input string.

        :type value: str
        :rytpe: str
        """
        if value is None:
            return None
        return percent_encode(value)

    def parse_url(self, value):
        """Parse a URL string into components.

        :type value: str
        :rtype: dict
        """
        if value is None:
            return None
        url_components = urlparse(value)
        try:
            url_components.port
        except ValueError:
            return None
        scheme = url_components.scheme
        query = url_components.query
        if scheme not in ('https', 'http') or len(query) > 0:
            return None
        path = url_components.path
        normalized_path = quote(normalize_url_path(path))
        if not normalized_path.endswith('/'):
            normalized_path = f'{normalized_path}/'
        return {'scheme': scheme, 'authority': url_components.netloc, 'path': path, 'normalizedPath': normalized_path, 'isIp': is_valid_ipv4_endpoint_url(value) or is_valid_ipv6_endpoint_url(value)}

    def boolean_equals(self, value1, value2):
        """Evaluates two boolean values for equality.

        :type value1: bool
        :type value2: bool
        :rtype: bool
        """
        if not all((isinstance(val, bool) for val in (value1, value2))):
            msg = f'Both arguments must be bools, not {type(value1)} and {type(value2)}.'
            raise EndpointResolutionError(msg=msg)
        return value1 is value2

    def is_ascii(self, value):
        """Evaluates if a string only contains ASCII characters.

        :type value: str
        :rtype: bool
        """
        try:
            value.encode('ascii')
            return True
        except UnicodeEncodeError:
            return False

    def substring(self, value, start, stop, reverse):
        """Computes a substring given the start index and end index. If `reverse` is
        True, slice the string from the end instead.

        :type value: str
        :type start: int
        :type end: int
        :type reverse: bool
        :rtype: str
        """
        if not isinstance(value, str):
            msg = f'Input must be a string, not {type(value)}.'
            raise EndpointResolutionError(msg=msg)
        if start >= stop or len(value) < stop or (not self.is_ascii(value)):
            return None
        if reverse is True:
            r_start = len(value) - stop
            r_stop = len(value) - start
            return value[r_start:r_stop]
        return value[start:stop]

    def _not(self, value):
        """A function implementation of the logical operator `not`.

        :type value: Any
        :rtype: bool
        """
        return not value

    def aws_is_virtual_hostable_s3_bucket(self, value, allow_subdomains):
        """Evaluates whether a value is a valid bucket name for virtual host
        style bucket URLs. To pass, the value must meet the following criteria:
        1. is_valid_host_label(value) is True
        2. length between 3 and 63 characters (inclusive)
        3. does not contain uppercase characters
        4. is not formatted as an IP address

        If allow_subdomains is True, split on `.` and validate
        each component separately.

        :type value: str
        :type allow_subdomains: bool
        :rtype: bool
        """
        if value is None or len(value) < 3 or value.lower() != value or (IPV4_RE.match(value) is not None):
            return False
        return self.is_valid_host_label(value, allow_subdomains=allow_subdomains)