from . import branch as _mod_branch
from . import controldir, errors
from . import revision as _mod_revision
from . import transport
from .i18n import gettext
from .trace import note, warning
def _show_push_branch(br_from, revision_id, location, to_file, verbose=False, overwrite=False, remember=False, stacked_on=None, create_prefix=False, use_existing_dir=False, no_tree=False, lossy=False):
    """Push a branch to a location.

    :param br_from: the source branch
    :param revision_id: the revision-id to push up to
    :param location: the url of the destination
    :param to_file: the output stream
    :param verbose: if True, display more output than normal
    :param overwrite: list of things to overwrite ("history", "tags")
        or boolean indicating for everything
    :param remember: if True, store the location as the push location for
        the source branch
    :param stacked_on: the url of the branch, if any, to stack on;
        if set, only the revisions not in that branch are pushed
    :param create_prefix: if True, create the necessary parent directories
        at the destination if they don't already exist
    :param use_existing_dir: if True, proceed even if the destination
        directory exists without a current control directory in it
    :param lossy: Allow lossy push
    """
    to_transport = transport.get_transport(location, purpose='write')
    try:
        dir_to = controldir.ControlDir.open_from_transport(to_transport)
    except errors.NotBranchError:
        dir_to = None
    if dir_to is None:
        try:
            br_to = br_from.create_clone_on_transport(to_transport, revision_id=revision_id, stacked_on=stacked_on, create_prefix=create_prefix, use_existing_dir=use_existing_dir, no_tree=no_tree)
        except errors.AlreadyControlDirError:
            raise errors.CommandError(gettext('Target directory %s already contains a .bzr directory, but it is not valid.') % (location,))
        except transport.FileExists:
            if not use_existing_dir:
                raise errors.CommandError(gettext('Target directory %s already exists, but does not have a .bzr directory. Supply --use-existing-dir to push there anyway.') % location)
            raise
        except transport.NoSuchFile:
            if not create_prefix:
                raise errors.CommandError(gettext('Parent directory of %s does not exist.\nYou may supply --create-prefix to create all leading parent directories.') % location)
            raise
        except errors.TooManyRedirections:
            raise errors.CommandError(gettext('Too many redirections trying to make %s.') % location)
        push_result = PushResult()
        try:
            push_result.stacked_on = br_to.get_stacked_on_url()
        except (_mod_branch.UnstackableBranchFormat, errors.UnstackableRepositoryFormat, errors.NotStacked):
            push_result.stacked_on = None
        push_result.target_branch = br_to
        push_result.old_revid = _mod_revision.NULL_REVISION
        push_result.old_revno = 0
        if remember or (remember is None and br_from.get_push_location() is None):
            br_from.set_push_location(br_to.base)
    else:
        if stacked_on is not None:
            warning('Ignoring request for a stacked branch as repository already exists at the destination location.')
        try:
            push_result = dir_to.push_branch(br_from, revision_id, overwrite, remember, create_prefix, lossy=lossy)
        except errors.DivergedBranches:
            raise errors.CommandError(gettext('These branches have diverged.  See "brz help diverged-branches" for more information.'))
        except errors.NoRoundtrippingSupport as e:
            raise errors.CommandError(gettext('It is not possible to losslessly push to %s. You may want to use --lossy.') % e.target_branch.mapping.vcs.abbreviation)
        except errors.NoRepositoryPresent:
            raise errors.CommandError(gettext('At %s you have a valid .bzr control directory, but not a branch or repository. This is an unsupported configuration. Please move the target directory out of the way and try again.') % location)
        if push_result.workingtree_updated is False:
            warning("This transport does not update the working tree of: %s. See 'brz help working-trees' for more information." % push_result.target_branch.base)
    push_result.report(to_file)
    if verbose:
        br_to = push_result.target_branch
        with br_to.lock_read():
            from .log import show_branch_change
            show_branch_change(br_to, to_file, push_result.old_revno, push_result.old_revid)