from __future__ import absolute_import, division, print_function
import json
from ansible.errors import AnsibleFilterError
Convert data which is in xml to json"
    :param data: The data passed in (data|from_xml(...))
    :type data: xml
    :param engine: Conversion library default=xml_to_dict
    