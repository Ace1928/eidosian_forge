from io import BytesIO
from ... import errors, lockable_files
from ...bzr.bzrdir import BzrDir, BzrDirFormat, BzrDirMetaFormat1
from ...controldir import (ControlDir, Converter, MustHaveWorkingTree,
from ...i18n import gettext
from ...lazy_import import lazy_import
from ...transport import NoSuchFile, get_transport, local
import os
from breezy import (
from breezy.bzr import (
from breezy.plugins.weave_fmt.store.versioned import VersionedFileStore
from breezy.transactions import WriteTransaction
from breezy.plugins.weave_fmt import xml4
class BzrDirFormatAllInOne(BzrDirFormat):
    """Common class for formats before meta-dirs."""
    fixed_components = True

    def initialize_on_transport_ex(self, transport, use_existing_dir=False, create_prefix=False, force_new_repo=False, stacked_on=None, stack_on_pwd=None, repo_format_name=None, make_working_trees=None, shared_repo=False):
        """See ControlDir.initialize_on_transport_ex."""
        require_stacking = stacked_on is not None
        if require_stacking:
            format = BzrDirMetaFormat1()
            return format.initialize_on_transport_ex(transport, use_existing_dir=use_existing_dir, create_prefix=create_prefix, force_new_repo=force_new_repo, stacked_on=stacked_on, stack_on_pwd=stack_on_pwd, repo_format_name=repo_format_name, make_working_trees=make_working_trees, shared_repo=shared_repo)
        return BzrDirFormat.initialize_on_transport_ex(self, transport, use_existing_dir=use_existing_dir, create_prefix=create_prefix, force_new_repo=force_new_repo, stacked_on=stacked_on, stack_on_pwd=stack_on_pwd, repo_format_name=repo_format_name, make_working_trees=make_working_trees, shared_repo=shared_repo)

    @classmethod
    def from_string(cls, format_string):
        if format_string != cls.get_format_string():
            raise AssertionError('unexpected format string %r' % format_string)
        return cls()