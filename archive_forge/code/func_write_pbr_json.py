import json
from pbr import git
def write_pbr_json(cmd, basename, filename):
    if not hasattr(cmd.distribution, 'pbr') or not cmd.distribution.pbr:
        return
    git_dir = git._run_git_functions()
    if not git_dir:
        return
    values = dict()
    git_version = git.get_git_short_sha(git_dir)
    is_release = git.get_is_release(git_dir)
    if git_version is not None:
        values['git_version'] = git_version
        values['is_release'] = is_release
        cmd.write_file('pbr', filename, json.dumps(values, sort_keys=True))