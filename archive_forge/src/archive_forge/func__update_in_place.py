from oslo_log import log as logging
from heat.common import exception
from heat.engine import dependencies
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.objects import resource as resource_objects
def _update_in_place(self, existing_res, new_res, is_substituted=False):
    existing_snippet = self.existing_snippets[existing_res.name]
    prev_res = self.previous_stack.get(new_res.name)
    new_snippet = new_res.t.reparse(self.existing_stack.defn, self.new_stack.t)
    if is_substituted:
        substitute = type(new_res)(existing_res.name, existing_res.t, existing_res.stack)
        existing_res.stack.resources[existing_res.name] = substitute
        existing_res = substitute
    existing_res.converge = self.new_stack.converge
    yield from existing_res.update(new_snippet, existing_snippet, prev_resource=prev_res)