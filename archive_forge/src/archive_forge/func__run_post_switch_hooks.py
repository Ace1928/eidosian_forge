from . import errors, lock, merge, revision
from .branch import Branch
from .i18n import gettext
from .trace import note
def _run_post_switch_hooks(control_dir, to_branch, force, revision_id):
    from .branch import SwitchHookParams
    hooks = Branch.hooks['post_switch']
    if not hooks:
        return
    params = SwitchHookParams(control_dir, to_branch, force, revision_id)
    for hook in hooks:
        hook(params)