import collections
import copy
import functools
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.heat import software_config as sc
from heat.engine.resources.openstack.heat import software_deployment as sd
from heat.engine import rsrc_defn
from heat.engine import support
class StructuredConfig(sc.SoftwareConfig):
    """A resource which has same logic with OS::Heat::SoftwareConfig.

    This resource is like OS::Heat::SoftwareConfig except that the config
    property is represented by a Map rather than a String.

    This is useful for configuration tools which use YAML or JSON as their
    configuration syntax. The resulting configuration is transferred,
    stored and returned by the software_configs API as parsed JSON.
    """
    support_status = support.SupportStatus(version='2014.1')
    PROPERTIES = GROUP, CONFIG, OPTIONS, INPUTS, OUTPUTS = (sc.SoftwareConfig.GROUP, sc.SoftwareConfig.CONFIG, sc.SoftwareConfig.OPTIONS, sc.SoftwareConfig.INPUTS, sc.SoftwareConfig.OUTPUTS)
    properties_schema = {GROUP: sc.SoftwareConfig.properties_schema[GROUP], OPTIONS: sc.SoftwareConfig.properties_schema[OPTIONS], INPUTS: sc.SoftwareConfig.properties_schema[INPUTS], OUTPUTS: sc.SoftwareConfig.properties_schema[OUTPUTS], CONFIG: properties.Schema(properties.Schema.MAP, _('Map representing the configuration data structure which will be serialized to JSON format.'))}