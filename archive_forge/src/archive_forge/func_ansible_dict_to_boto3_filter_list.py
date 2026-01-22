from ansible.module_utils.six import integer_types
from ansible.module_utils.six import string_types
def ansible_dict_to_boto3_filter_list(filters_dict):
    """Convert an Ansible dict of filters to list of dicts that boto3 can use
    Args:
        filters_dict (dict): Dict of AWS filters.
    Basic Usage:
        >>> filters = {'some-aws-id': 'i-01234567'}
        >>> ansible_dict_to_boto3_filter_list(filters)
        {
            'some-aws-id': 'i-01234567'
        }
    Returns:
        List: List of AWS filters and their values
        [
            {
                'Name': 'some-aws-id',
                'Values': [
                    'i-01234567',
                ]
            }
        ]
    """
    filters_list = []
    for k, v in filters_dict.items():
        filter_dict = {'Name': k}
        if isinstance(v, bool):
            filter_dict['Values'] = [str(v).lower()]
        elif isinstance(v, integer_types):
            filter_dict['Values'] = [str(v)]
        elif isinstance(v, string_types):
            filter_dict['Values'] = [v]
        else:
            filter_dict['Values'] = v
        filters_list.append(filter_dict)
    return filters_list