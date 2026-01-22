from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from ruamel import yaml
import six
def LocationValueLoad(source):
    """Loads location/value objects from YAML source.

  Call this indirectly by:

    core.yaml.load(source, location_value=True)

  Args:
    source: A file like object or string containing YAML data.

  Returns:
    The YAML data, where each data item is an object with value and lc
    attributes, where lc.line and lc.col are the line and column location for
    the item in the YAML source file.
  """
    yml = yaml.YAML()
    yml.Constructor = _LvObjectConstructor
    return yml.load(source)