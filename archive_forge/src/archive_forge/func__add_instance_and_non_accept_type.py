from collections import namedtuple
import re
import textwrap
import warnings
def _add_instance_and_non_accept_type(self, instance, other, instance_on_the_right=False):
    if other is None:
        return AcceptNoHeader()
    other_header_value = self._python_value_to_header_str(value=other)
    try:
        return AcceptValidHeader(header_value=other_header_value)
    except ValueError:
        return AcceptNoHeader()