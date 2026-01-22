from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
from ansible_collections.community.general.plugins.module_utils.gconftool2 import gconftool2_runner
class GConftool(StateModuleHelper):
    diff_params = ('value',)
    output_params = ('key', 'value_type')
    facts_params = ('key', 'value_type')
    facts_name = 'gconftool2'
    module = dict(argument_spec=dict(key=dict(type='str', required=True, no_log=False), value_type=dict(type='str', choices=['bool', 'float', 'int', 'string']), value=dict(type='str'), state=dict(type='str', required=True, choices=['absent', 'present']), direct=dict(type='bool', default=False), config_source=dict(type='str')), required_if=[('state', 'present', ['value', 'value_type']), ('direct', True, ['config_source'])], supports_check_mode=True)

    def __init_module__(self):
        self.runner = gconftool2_runner(self.module, check_rc=True)
        if self.vars.state != 'get':
            if not self.vars.direct and self.vars.config_source is not None:
                self.module.fail_json(msg='If the "config_source" is specified then "direct" must be "true"')
        self.vars.set('previous_value', self._get(), fact=True)
        self.vars.set('value_type', self.vars.value_type)
        self.vars.set('_value', self.vars.previous_value, output=False, change=True)
        self.vars.set_meta('value', initial_value=self.vars.previous_value)
        self.vars.set('playbook_value', self.vars.value, fact=True)

    def _make_process(self, fail_on_err):

        def process(rc, out, err):
            if err and fail_on_err:
                self.ansible.fail_json(msg='gconftool-2 failed with error: %s' % str(err))
            out = out.rstrip()
            self.vars.value = None if out == '' else out
            return self.vars.value
        return process

    def _get(self):
        return self.runner('state key', output_process=self._make_process(False)).run(state='get')

    def state_absent(self):
        with self.runner('state key', output_process=self._make_process(False)) as ctx:
            ctx.run()
            if self.verbosity >= 4:
                self.vars.run_info = ctx.run_info
        self.vars.set('new_value', None, fact=True)
        self.vars._value = None

    def state_present(self):
        with self.runner('direct config_source value_type state key value', output_process=self._make_process(True)) as ctx:
            ctx.run()
            if self.verbosity >= 4:
                self.vars.run_info = ctx.run_info
        self.vars.set('new_value', self._get(), fact=True)
        self.vars._value = self.vars.new_value