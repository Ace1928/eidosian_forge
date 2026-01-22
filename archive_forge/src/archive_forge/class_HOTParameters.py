from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints as constr
from heat.engine import parameters
class HOTParameters(parameters.Parameters):
    PSEUDO_PARAMETERS = PARAM_STACK_ID, PARAM_STACK_NAME, PARAM_REGION, PARAM_PROJECT_ID = ('OS::stack_id', 'OS::stack_name', 'OS::region', 'OS::project_id')

    def set_stack_id(self, stack_identifier):
        """Set the StackId pseudo parameter value."""
        if stack_identifier is not None:
            self.params[self.PARAM_STACK_ID].schema.set_default(stack_identifier.stack_id)
            return True
        return False

    def _pseudo_parameters(self, stack_identifier):
        stack_id = getattr(stack_identifier, 'stack_id', '')
        stack_name = getattr(stack_identifier, 'stack_name', '')
        tenant = getattr(stack_identifier, 'tenant', '')
        yield parameters.Parameter(self.PARAM_STACK_ID, parameters.Schema(parameters.Schema.STRING, _('Stack ID'), default=str(stack_id)))
        yield parameters.Parameter(self.PARAM_PROJECT_ID, parameters.Schema(parameters.Schema.STRING, _('Project ID'), default=str(tenant)))
        if stack_name:
            yield parameters.Parameter(self.PARAM_STACK_NAME, parameters.Schema(parameters.Schema.STRING, _('Stack Name'), default=stack_name))