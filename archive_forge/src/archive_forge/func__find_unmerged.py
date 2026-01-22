from . import errors, log
def _find_unmerged(local_branch, remote_branch, restrict, include_merged, backward, local_revid_range=None, remote_revid_range=None):
    """See find_unmerged.

    The branches should already be locked before entering.
    """
    try:
        local_revno, local_revision_id = local_branch.last_revision_info()
    except (errors.UnsupportedOperation, errors.GhostRevisionsHaveNoRevno):
        local_revno = None
        local_revision_id = local_branch.last_revision()
    try:
        remote_revno, remote_revision_id = remote_branch.last_revision_info()
    except (errors.UnsupportedOperation, errors.GhostRevisionsHaveNoRevno):
        remote_revision_id = remote_branch.last_revision()
        remote_revno = None
    if local_revision_id == remote_revision_id:
        return ([], [])
    graph = local_branch.repository.get_graph(remote_branch.repository)
    if restrict == 'remote':
        local_extra = None
        remote_extra = graph.find_unique_ancestors(remote_revision_id, [local_revision_id])
    elif restrict == 'local':
        remote_extra = None
        local_extra = graph.find_unique_ancestors(local_revision_id, [remote_revision_id])
    else:
        if restrict != 'all':
            raise ValueError('param restrict not one of "all", "local", "remote": %r' % (restrict,))
        local_extra, remote_extra = graph.find_difference(local_revision_id, remote_revision_id)
    if include_merged:
        locals = _enumerate_with_merges(local_branch, local_extra, graph, local_revno, local_revision_id, backward)
        remotes = _enumerate_with_merges(remote_branch, remote_extra, graph, remote_revno, remote_revision_id, backward)
    else:
        locals = _enumerate_mainline(local_extra, graph, local_revno, local_revision_id, backward)
        remotes = _enumerate_mainline(remote_extra, graph, remote_revno, remote_revision_id, backward)
    return (_filter_revs(graph, locals, local_revid_range), _filter_revs(graph, remotes, remote_revid_range))