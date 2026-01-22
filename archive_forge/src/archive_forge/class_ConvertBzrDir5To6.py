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
class ConvertBzrDir5To6(Converter):
    """Converts format 5 bzr dirs to format 6."""

    def convert(self, to_convert, pb):
        """See Converter.convert()."""
        self.controldir = to_convert
        with ui.ui_factory.nested_progress_bar() as pb:
            ui.ui_factory.note(gettext('starting upgrade from format 5 to 6'))
            self._convert_to_prefixed()
            return ControlDir.open(self.controldir.user_url)

    def _convert_to_prefixed(self):
        from .store import TransportStore
        self.controldir.transport.delete('branch-format')
        for store_name in ['weaves', 'revision-store']:
            ui.ui_factory.note(gettext('adding prefixes to %s') % store_name)
            store_transport = self.controldir.transport.clone(store_name)
            store = TransportStore(store_transport, prefixed=True)
            for urlfilename in store_transport.list_dir('.'):
                filename = urlutils.unescape(urlfilename)
                if filename.endswith('.weave') or filename.endswith('.gz') or filename.endswith('.sig'):
                    file_id, suffix = os.path.splitext(filename)
                else:
                    file_id = filename
                    suffix = ''
                new_name = store._mapper.map((file_id.encode('utf-8'),)) + suffix
                try:
                    store_transport.move(filename, new_name)
                except NoSuchFile:
                    store_transport.mkdir(osutils.dirname(new_name))
                    store_transport.move(filename, new_name)
        self.controldir.transport.put_bytes('branch-format', BzrDirFormat6().get_format_string(), mode=self.controldir._get_file_mode())