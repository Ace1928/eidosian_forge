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
class StructuredDeploymentGroup(sd.SoftwareDeploymentGroup):
    """This resource associates a group of servers with some configuration.

    This resource works similar as OS::Heat::SoftwareDeploymentGroup, but for
    structured resources.
    """
    PROPERTIES = SERVERS, CONFIG, INPUT_VALUES, DEPLOY_ACTIONS, NAME, SIGNAL_TRANSPORT, INPUT_KEY, INPUT_VALUES_VALIDATE = (sd.SoftwareDeploymentGroup.SERVERS, sd.SoftwareDeploymentGroup.CONFIG, sd.SoftwareDeploymentGroup.INPUT_VALUES, sd.SoftwareDeploymentGroup.DEPLOY_ACTIONS, sd.SoftwareDeploymentGroup.NAME, sd.SoftwareDeploymentGroup.SIGNAL_TRANSPORT, StructuredDeployment.INPUT_KEY, StructuredDeployment.INPUT_VALUES_VALIDATE)
    _sds_ps = sd.SoftwareDeploymentGroup.properties_schema
    properties_schema = {SERVERS: _sds_ps[SERVERS], CONFIG: _sds_ps[CONFIG], INPUT_VALUES: _sds_ps[INPUT_VALUES], DEPLOY_ACTIONS: _sds_ps[DEPLOY_ACTIONS], SIGNAL_TRANSPORT: _sds_ps[SIGNAL_TRANSPORT], NAME: _sds_ps[NAME], INPUT_KEY: StructuredDeployment.properties_schema[INPUT_KEY], INPUT_VALUES_VALIDATE: StructuredDeployment.properties_schema[INPUT_VALUES_VALIDATE]}

    def build_resource_definition(self, res_name, res_defn):
        props = copy.deepcopy(res_defn)
        servers = props.pop(self.SERVERS)
        props[StructuredDeployment.SERVER] = servers.get(res_name)
        return rsrc_defn.ResourceDefinition(res_name, 'OS::Heat::StructuredDeployment', props, None)