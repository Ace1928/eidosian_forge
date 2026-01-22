import breezy.bzr  # noqa: F401
from breezy import controldir
from ..commands import Command, display_command
from ..option import Option, RegistryOption
class cmd_git_import(Command):
    """Import all branches from a git repository.

    """
    takes_args = ['src_location', 'dest_location?']
    takes_options = [Option('colocated', help='Create colocated branches.'), RegistryOption('dest-format', help='Specify a format for this branch. See "help formats" for a full list.', lazy_registry=('breezy.controldir', 'format_registry'), converter=lambda name: controldir.format_registry.make_controldir(name), value_switches=True, title='Branch format')]

    def _get_colocated_branch(self, target_controldir, name):
        from ..errors import NotBranchError
        try:
            return target_controldir.open_branch(name=name)
        except NotBranchError:
            return target_controldir.create_branch(name=name)

    def _get_nested_branch(self, dest_transport, dest_format, name):
        from ..controldir import ControlDir
        from ..errors import NotBranchError
        head_transport = dest_transport.clone(name)
        try:
            head_controldir = ControlDir.open_from_transport(head_transport)
        except NotBranchError:
            head_controldir = dest_format.initialize_on_transport_ex(head_transport, create_prefix=True)[1]
        try:
            return head_controldir.open_branch()
        except NotBranchError:
            return head_controldir.create_branch()

    def run(self, src_location, dest_location=None, colocated=False, dest_format=None):
        import os
        from .. import controldir, trace, ui, urlutils
        from ..controldir import ControlDir
        from ..errors import BzrError, CommandError, NoRepositoryPresent, NotBranchError
        from ..i18n import gettext
        from ..repository import InterRepository, Repository
        from ..transport import get_transport
        from .branch import LocalGitBranch
        from .refs import ref_to_branch_name
        from .repository import GitRepository
        if dest_format is None:
            dest_format = controldir.format_registry.make_controldir('default')
        if dest_location is None:
            dest_location = os.path.basename(src_location.rstrip('/\\'))
        dest_transport = get_transport(dest_location)
        source_repo = Repository.open(src_location)
        if not isinstance(source_repo, GitRepository):
            raise CommandError(gettext('%r is not a git repository') % src_location)
        try:
            target_controldir = ControlDir.open_from_transport(dest_transport)
        except NotBranchError:
            target_controldir = dest_format.initialize_on_transport_ex(dest_transport, shared_repo=True)[1]
        try:
            target_repo = target_controldir.find_repository()
        except NoRepositoryPresent:
            target_repo = target_controldir.create_repository(shared=True)
        if not target_repo.supports_rich_root():
            raise CommandError(gettext("Target repository doesn't support rich roots"))
        interrepo = InterRepository.get(source_repo, target_repo)
        mapping = source_repo.get_mapping()
        result = interrepo.fetch()
        with ui.ui_factory.nested_progress_bar() as pb:
            for i, (name, sha) in enumerate(result.refs.items()):
                try:
                    branch_name = ref_to_branch_name(name)
                except ValueError:
                    continue
                pb.update(gettext('creating branches'), i, len(result.refs))
                if getattr(target_controldir._format, 'colocated_branches', False) and colocated:
                    if name == 'HEAD':
                        branch_name = None
                    head_branch = self._get_colocated_branch(target_controldir, branch_name)
                else:
                    head_branch = self._get_nested_branch(dest_transport, dest_format, branch_name)
                revid = mapping.revision_id_foreign_to_bzr(sha)
                source_branch = LocalGitBranch(source_repo.controldir, source_repo, sha)
                if head_branch.last_revision() != revid:
                    head_branch.generate_revision_history(revid)
                source_branch.tags.merge_to(head_branch.tags)
                if not head_branch.get_parent():
                    url = urlutils.join_segment_parameters(source_branch.base, {'branch': urlutils.escape(branch_name)})
                    head_branch.set_parent(url)
        trace.note(gettext("Use 'bzr checkout' to create a working tree in the newly created branches."))