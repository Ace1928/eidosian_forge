from oslo_config import cfg
class SchedulerHintsMixin(object):
    """Utility class to encapsulate Scheduler Hint related logic."""
    HEAT_ROOT_STACK_ID = 'heat_root_stack_id'
    HEAT_STACK_ID = 'heat_stack_id'
    HEAT_STACK_NAME = 'heat_stack_name'
    HEAT_PATH_IN_STACK = 'heat_path_in_stack'
    HEAT_RESOURCE_NAME = 'heat_resource_name'
    HEAT_RESOURCE_UUID = 'heat_resource_uuid'

    @staticmethod
    def _path_in_stack(stack):
        path = []
        for parent_res_name, stack_name in stack.path_in_stack():
            if parent_res_name is not None:
                path.append(','.join([parent_res_name, stack_name]))
            else:
                path.append(stack_name)
        return path

    def _scheduler_hints(self, scheduler_hints):
        """Augment scheduler hints with supplemental content."""
        if cfg.CONF.stack_scheduler_hints:
            if scheduler_hints is None:
                scheduler_hints = {}
            stack = self.stack
            scheduler_hints[self.HEAT_ROOT_STACK_ID] = stack.root_stack_id()
            scheduler_hints[self.HEAT_STACK_ID] = stack.id
            scheduler_hints[self.HEAT_STACK_NAME] = stack.name
            scheduler_hints[self.HEAT_PATH_IN_STACK] = self._path_in_stack(stack)
            scheduler_hints[self.HEAT_RESOURCE_NAME] = self.name
            scheduler_hints[self.HEAT_RESOURCE_UUID] = self.uuid
        return scheduler_hints