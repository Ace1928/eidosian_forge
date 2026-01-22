from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleLookupError
from ansible.module_utils.common._collections_compat import Mapping, Sequence
from ansible.module_utils.six import string_types
from ansible.plugins.lookup import LookupBase
from ansible.release import __version__ as ansible_version
from ansible.template import Templar
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def __process(self, result, terms, index, current, templar, variables):
    """Fills ``result`` list with evaluated items.

        ``result`` is a list where the resulting items are placed.
        ``terms`` is the parsed list of terms
        ``index`` is the current index to be processed in the list.
        ``current`` is a dictionary where the first ``index`` values are filled in.
        ``variables`` are the variables currently available.
        """
    if index == len(terms):
        result.append(current.copy())
        return
    key, expression, values = terms[index]
    if expression is not None:
        vars = variables.copy()
        vars['item'] = current.copy()
        try:
            values = self.__evaluate(expression, templar, variables=vars)
        except Exception as e:
            raise AnsibleLookupError('Caught "{error}" while evaluating {key!r} with item == {item!r}'.format(error=e, key=key, item=current))
    if isinstance(values, Mapping):
        for idx, val in sorted(values.items()):
            current[key] = dict([('key', idx), ('value', val)])
            self.__process(result, terms, index + 1, current, templar, variables)
    elif isinstance(values, Sequence):
        for elt in values:
            current[key] = elt
            self.__process(result, terms, index + 1, current, templar, variables)
    else:
        raise AnsibleLookupError('Did not obtain dictionary or list while evaluating {key!r} with item == {item!r}, but {type}'.format(key=key, item=current, type=type(values)))