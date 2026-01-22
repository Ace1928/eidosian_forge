import collections
import weakref
from heat.common import context
from heat.common import exception
from heat.common.i18n import _
from heat.db import api as db_api
from heat.objects import raw_template_files
class TemplateFiles(collections.abc.Mapping):

    def __init__(self, files):
        self.files = None
        self.files_id = None
        if files is None:
            return
        if isinstance(files, TemplateFiles):
            self.files_id = files.files_id
            self.files = files.files
            return
        if isinstance(files, int):
            self.files_id = files
            if self.files_id in _d:
                self.files = _d[self.files_id]
            return
        if not isinstance(files, dict):
            raise ValueError(_('Expected dict, got %(cname)s for files, (value is %(val)s)') % {'cname': files.__class__, 'val': str(files)})
        self.files = ReadOnlyDict(files)

    def __getitem__(self, key):
        self._refresh_if_needed()
        if self.files is None:
            raise KeyError
        return self.files[key]

    def __setitem__(self, key, value):
        self.update({key: value})

    def __len__(self):
        self._refresh_if_needed()
        if not self.files:
            return 0
        return len(self.files)

    def __contains__(self, key):
        self._refresh_if_needed()
        if not self.files:
            return False
        return key in self.files

    def __iter__(self):
        self._refresh_if_needed()
        if self.files is None:
            return iter(ReadOnlyDict({}))
        return iter(self.files)

    def _refresh_if_needed(self):
        if self.files_id is None:
            return
        if self.files_id in _d:
            self.files = _d[self.files_id]
            return
        self._refresh()

    def _refresh(self):
        ctxt = context.get_admin_context()
        rtf_obj = db_api.raw_template_files_get(ctxt, self.files_id)
        _files_dict = ReadOnlyDict(rtf_obj.files)
        self.files = _files_dict
        _d[self.files_id] = _files_dict

    def store(self, ctxt):
        if not self.files or self.files_id is not None:
            return self.files_id
        rtf_obj = raw_template_files.RawTemplateFiles.create(ctxt, {'files': self.files})
        self.files_id = rtf_obj.id
        _d[self.files_id] = self.files
        return self.files_id

    def update(self, files):
        if len(files) == 0:
            return
        if not isinstance(files, dict):
            raise ValueError(_('Expected dict, got %(cname)s for files, (value is %(val)s)') % {'cname': files.__class__, 'val': str(files)})
        self._refresh_if_needed()
        if self.files:
            new_files = self.files.copy()
            new_files.update(files)
        else:
            new_files = files
        self.files_id = None
        self.files = ReadOnlyDict(new_files)