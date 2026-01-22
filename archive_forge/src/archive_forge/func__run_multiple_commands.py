from __future__ import absolute_import, division, print_function
import re
import json
import numbers
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
from ansible_collections.community.general.plugins.module_utils.snap import snap_runner
def _run_multiple_commands(self, commands, actionable_names, bundle=True, refresh=False):
    results_cmd = []
    results_rc = []
    results_out = []
    results_err = []
    results_run_info = []
    state = 'refresh' if refresh else self.vars.state
    with self.runner(commands + ['name']) as ctx:
        if bundle:
            rc, out, err = ctx.run(state=state, name=actionable_names)
            results_cmd.append(commands + actionable_names)
            results_rc.append(rc)
            results_out.append(out.strip())
            results_err.append(err.strip())
            results_run_info.append(ctx.run_info)
        else:
            for name in actionable_names:
                rc, out, err = ctx.run(state=state, name=name)
                results_cmd.append(commands + [name])
                results_rc.append(rc)
                results_out.append(out.strip())
                results_err.append(err.strip())
                results_run_info.append(ctx.run_info)
    return ('; '.join([to_native(x) for x in results_cmd]), self._first_non_zero(results_rc), '\n'.join(results_out), '\n'.join(results_err), results_run_info)