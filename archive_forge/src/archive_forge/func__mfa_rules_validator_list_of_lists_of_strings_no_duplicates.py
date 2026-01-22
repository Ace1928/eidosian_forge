from keystone.common import resource_options
from keystone.common.validation import parameter_types
from keystone.i18n import _
def _mfa_rules_validator_list_of_lists_of_strings_no_duplicates(value):
    msg = _('Invalid data type, must be a list of lists comprised of strings. Sub-lists may not be duplicated. Strings in sub-lists may not be duplicated.')
    if not isinstance(value, list):
        raise TypeError(msg)
    sublists = []
    for sublist in value:
        string_set = set()
        if not isinstance(sublist, list):
            raise TypeError(msg)
        if not sublist:
            raise ValueError(msg)
        if sublist in sublists:
            raise ValueError(msg)
        sublists.append(sublist)
        for element in sublist:
            if not isinstance(element, str):
                raise TypeError(msg)
            if element in string_set:
                raise ValueError(msg)
            string_set.add(element)