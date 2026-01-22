from datetime import datetime
from oslo_utils import timeutils
class UnsupportedAttribute(AttributeError):
    """Indicates that the user is trying to transmit the argument to a method,
    which is not supported by selected version.
    """

    def __init__(self, argument_name, start_version, end_version):
        if start_version and end_version:
            self.message = "'%(name)s' argument is only allowed for microversions %(start)s - %(end)s." % {'name': argument_name, 'start': start_version.get_string(), 'end': end_version.get_string()}
        elif start_version:
            self.message = "'%(name)s' argument is only allowed since microversion %(start)s." % {'name': argument_name, 'start': start_version.get_string()}
        elif end_version:
            self.message = "'%(name)s' argument is not allowed after microversion %(end)s." % {'name': argument_name, 'end': end_version.get_string()}

    def __str__(self):
        return self.message