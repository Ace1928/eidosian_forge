from operator import itemgetter
from breezy import controldir
from ... import errors, osutils, transport
from ...trace import note, show_error
from .helpers import best_format_for_objects_in_a_repository, single_plural
def _get_matching_branches(self):
    """Get the Bazaar branches.

        :return: branch_tips, lost_heads where
          branch_tips = a list of (branch,tip) tuples for branches. The
            first tip is the 'trunk'.
          lost_heads = a list of (bazaar-name,revision) for branches that
            would have been created had the repository been shared and
            everything succeeded
        """
    branch_tips = []
    lost_heads = []
    ref_names = list(self.heads_by_ref)
    if self.branch is not None:
        trunk = self.select_trunk(ref_names)
        default_tip = self.heads_by_ref[trunk][0]
        branch_tips.append((self.branch, default_tip))
        ref_names.remove(trunk)
    git_to_bzr_map = {}
    for ref_name in ref_names:
        git_to_bzr_map[ref_name] = self.cache_mgr.branch_mapper.git_to_bzr(ref_name)
    if ref_names and self.branch is None:
        trunk = self.select_trunk(ref_names)
        git_bzr_items = [(trunk, git_to_bzr_map[trunk])]
        del git_to_bzr_map[trunk]
    else:
        git_bzr_items = []
    git_bzr_items.extend(sorted(git_to_bzr_map.items(), key=itemgetter(1)))

    def dir_under_current(name):
        repo_base = self.repo.controldir.transport.base
        return osutils.pathjoin(repo_base, '..', name)

    def dir_sister_branch(name):
        return osutils.pathjoin(self.branch.base, '..', name)
    if self.branch is not None:
        dir_policy = dir_sister_branch
    else:
        dir_policy = dir_under_current
    can_create_branches = self.repo.is_shared() or self.repo.controldir._format.colocated_branches
    for ref_name, name in git_bzr_items:
        tip = self.heads_by_ref[ref_name][0]
        if can_create_branches:
            try:
                br = self.make_branch(name, ref_name, dir_policy)
                branch_tips.append((br, tip))
                continue
            except errors.BzrError as ex:
                show_error('ERROR: failed to create branch %s: %s', name, ex)
        lost_head = self.cache_mgr.lookup_committish(tip)
        lost_info = (name, lost_head)
        lost_heads.append(lost_info)
    return (branch_tips, lost_heads)