from .base import Submodule, UpdateProgress
from .util import find_first_remote_branch
from git.exc import InvalidGitRepositoryError
import git
import logging
from typing import TYPE_CHECKING, Union
from git.types import Commit_ish
class RootModule(Submodule):
    """A (virtual) root of all submodules in the given repository.

    This can be used to more easily traverse all submodules of the
    superproject (master repository).
    """
    __slots__ = ()
    k_root_name = '__ROOT__'

    def __init__(self, repo: 'Repo'):
        super().__init__(repo, binsha=self.NULL_BIN_SHA, mode=self.k_default_mode, path='', name=self.k_root_name, parent_commit=repo.head.commit, url='', branch_path=git.Head.to_full_path(self.k_head_default))

    def _clear_cache(self) -> None:
        """May not do anything."""
        pass

    def update(self, previous_commit: Union[Commit_ish, None]=None, recursive: bool=True, force_remove: bool=False, init: bool=True, to_latest_revision: bool=False, progress: Union[None, 'RootUpdateProgress']=None, dry_run: bool=False, force_reset: bool=False, keep_going: bool=False) -> 'RootModule':
        """Update the submodules of this repository to the current HEAD commit.

        This method behaves smartly by determining changes of the path of a submodules
        repository, next to changes to the to-be-checked-out commit or the branch to be
        checked out. This works if the submodules ID does not change.
        Additionally it will detect addition and removal of submodules, which will be
        handled gracefully.

        :param previous_commit: If set to a commit-ish, the commit we should use as the
            previous commit the HEAD pointed to before it was set to the commit it
            points to now.
            If None, it defaults to ``HEAD@{1}`` otherwise
        :param recursive: if True, the children of submodules will be updated as well
            using the same technique.
        :param force_remove: If submodules have been deleted, they will be forcibly
            removed. Otherwise the update may fail if a submodule's repository cannot be
            deleted as changes have been made to it.
            (See :meth:`Submodule.update <git.objects.submodule.base.Submodule.update>`
            for more information.)
        :param init: If we encounter a new module which would need to be initialized,
            then do it.
        :param to_latest_revision: If True, instead of checking out the revision pointed
            to by this submodule's sha, the checked out tracking branch will be merged
            with the latest remote branch fetched from the repository's origin.
            Unless `force_reset` is specified, a local tracking branch will never be
            reset into its past, therefore the remote branch must be in the future for
            this to have an effect.
        :param force_reset: If True, submodules may checkout or reset their branch even
            if the repository has pending changes that would be overwritten, or if the
            local tracking branch is in the future of the remote tracking branch and
            would be reset into its past.
        :param progress: :class:`RootUpdateProgress` instance or None if no progress
            should be sent.
        :param dry_run: If True, operations will not actually be performed. Progress
            messages will change accordingly to indicate the WOULD DO state of the
            operation.
        :param keep_going: If True, we will ignore but log all errors, and keep going
            recursively. Unless `dry_run` is set as well, `keep_going` could cause
            subsequent/inherited errors you wouldn't see otherwise.
            In conjunction with `dry_run`, this can be useful to anticipate all errors
            when updating submodules.
        :return: self
        """
        if self.repo.bare:
            raise InvalidGitRepositoryError('Cannot update submodules in bare repositories')
        if progress is None:
            progress = RootUpdateProgress()
        prefix = ''
        if dry_run:
            prefix = 'DRY-RUN: '
        repo = self.repo
        try:
            cur_commit = repo.head.commit
            if previous_commit is None:
                try:
                    previous_commit = repo.commit(repo.head.log_entry(-1).oldhexsha)
                    if previous_commit.binsha == previous_commit.NULL_BIN_SHA:
                        raise IndexError
                except IndexError:
                    previous_commit = cur_commit
            else:
                previous_commit = repo.commit(previous_commit)
            psms: 'IterableList[Submodule]' = self.list_items(repo, parent_commit=previous_commit)
            sms: 'IterableList[Submodule]' = self.list_items(repo)
            spsms = set(psms)
            ssms = set(sms)
            rrsm = spsms - ssms
            len_rrsm = len(rrsm)
            for i, rsm in enumerate(rrsm):
                op = REMOVE
                if i == 0:
                    op |= BEGIN
                progress.update(op, i, len_rrsm, prefix + 'Removing submodule %r at %s' % (rsm.name, rsm.abspath))
                rsm._parent_commit = repo.head.commit
                rsm.remove(configuration=False, module=True, force=force_remove, dry_run=dry_run)
                if i == len_rrsm - 1:
                    op |= END
                progress.update(op, i, len_rrsm, prefix + 'Done removing submodule %r' % rsm.name)
            csms = spsms & ssms
            len_csms = len(csms)
            for i, csm in enumerate(csms):
                psm: 'Submodule' = psms[csm.name]
                sm: 'Submodule' = sms[csm.name]
                if sm.path != psm.path and psm.module_exists():
                    progress.update(BEGIN | PATHCHANGE, i, len_csms, prefix + 'Moving repository of submodule %r from %s to %s' % (sm.name, psm.abspath, sm.abspath))
                    if not dry_run:
                        psm.move(sm.path, module=True, configuration=False)
                    progress.update(END | PATHCHANGE, i, len_csms, prefix + 'Done moving repository of submodule %r' % sm.name)
                if sm.module_exists():
                    if sm.url != psm.url:
                        nn = '__new_origin__'
                        smm = sm.module()
                        rmts = smm.remotes
                        if len([r for r in rmts if r.url == sm.url]) == 0:
                            progress.update(BEGIN | URLCHANGE, i, len_csms, prefix + 'Changing url of submodule %r from %s to %s' % (sm.name, psm.url, sm.url))
                            if not dry_run:
                                assert nn not in [r.name for r in rmts]
                                smr = smm.create_remote(nn, sm.url)
                                smr.fetch(progress=progress)
                                if len([r for r in smr.refs if r.remote_head == sm.branch_name]) == 0:
                                    raise ValueError('Submodule branch named %r was not available in new submodule remote at %r' % (sm.branch_name, sm.url))
                                rmt_for_deletion = None
                                for remote in rmts:
                                    if remote.url == psm.url:
                                        rmt_for_deletion = remote
                                        break
                                if rmt_for_deletion is None:
                                    if len(rmts) == 1:
                                        rmt_for_deletion = rmts[0]
                                    else:
                                        raise InvalidGitRepositoryError("Couldn't find original remote-repo at url %r" % psm.url)
                                orig_name = rmt_for_deletion.name
                                smm.delete_remote(rmt_for_deletion)
                                smr.rename(orig_name)
                                smsha = sm.binsha
                                found = False
                                rref = smr.refs[self.branch_name]
                                for c in rref.commit.traverse():
                                    if c.binsha == smsha:
                                        found = True
                                        break
                                if not found:
                                    _logger.warning("Current sha %s was not contained in the tracking             branch at the new remote, setting it the the remote's tracking branch", sm.hexsha)
                                    sm.binsha = rref.commit.binsha
                            progress.update(END | URLCHANGE, i, len_csms, prefix + 'Done adjusting url of submodule %r' % sm.name)
                    if sm.branch_path != psm.branch_path:
                        progress.update(BEGIN | BRANCHCHANGE, i, len_csms, prefix + 'Changing branch of submodule %r from %s to %s' % (sm.name, psm.branch_path, sm.branch_path))
                        if not dry_run:
                            smm = sm.module()
                            smmr = smm.remotes
                            for remote in smmr:
                                remote.fetch(progress=progress)
                            try:
                                tbr = git.Head.create(smm, sm.branch_name, logmsg='branch: Created from HEAD')
                            except OSError:
                                tbr = git.Head(smm, sm.branch_path)
                            tbr.set_tracking_branch(find_first_remote_branch(smmr, sm.branch_name))
                            smm.head.reference = tbr
                        progress.update(END | BRANCHCHANGE, i, len_csms, prefix + 'Done changing branch of submodule %r' % sm.name)
        except Exception as err:
            if not keep_going:
                raise
            _logger.error(str(err))
        for sm in sms:
            sm.update(recursive=False, init=init, to_latest_revision=to_latest_revision, progress=progress, dry_run=dry_run, force=force_reset, keep_going=keep_going)
            if recursive:
                if sm.module_exists():
                    type(self)(sm.module()).update(recursive=True, force_remove=force_remove, init=init, to_latest_revision=to_latest_revision, progress=progress, dry_run=dry_run, force_reset=force_reset, keep_going=keep_going)
        return self

    def module(self) -> 'Repo':
        """:return: The actual repository containing the submodules"""
        return self.repo