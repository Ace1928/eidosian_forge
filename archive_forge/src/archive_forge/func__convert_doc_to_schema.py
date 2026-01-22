from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.utils.plugins.module_utils.common.utils import dict_merge
def _convert_doc_to_schema(self):
    """Convert the doc string to an obj, was yaml
        add back other valid conditionals and params
        """
    doc_obj = yaml.load(self._schema, SafeLoader)
    temp_schema = {}
    self._extract_schema_from_doc(doc_obj, temp_schema)
    self._schema = {'argument_spec': temp_schema}