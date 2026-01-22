from oslo_config import cfg
from taskflow.patterns import linear_flow as lf
from taskflow import task
from glance.i18n import _
class _InjectMetadataProperties(task.Task):

    def __init__(self, context, task_id, task_type, action_wrapper):
        self.context = context
        self.task_id = task_id
        self.task_type = task_type
        self.action_wrapper = action_wrapper
        self.image_id = action_wrapper.image_id
        super(_InjectMetadataProperties, self).__init__(name='%s-InjectMetadataProperties-%s' % (task_type, task_id))

    def execute(self):
        """Inject custom metadata properties to image

        :param image_id: Glance Image ID
        """
        user_roles = self.context.roles
        ignore_user_roles = CONF.inject_metadata_properties.ignore_user_roles
        if not [role for role in user_roles if role in ignore_user_roles]:
            properties = CONF.inject_metadata_properties.inject
            if properties:
                with self.action_wrapper as action:
                    action.set_image_extra_properties(properties)