from keystoneauth1 import exceptions
from keystoneauth1.identity import v3
@staticmethod
def xml_to_str(content, **kwargs):
    return etree.tostring(content, **kwargs)