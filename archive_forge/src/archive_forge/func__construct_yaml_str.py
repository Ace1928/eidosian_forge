import yaml
from oslo_serialization import jsonutils
def _construct_yaml_str(self, node):
    return self.construct_scalar(node)