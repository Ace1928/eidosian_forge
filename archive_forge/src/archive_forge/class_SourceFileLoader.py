import _imp
import _io
import sys
import _warnings
import marshal
class SourceFileLoader(FileLoader, SourceLoader):
    """Concrete implementation of SourceLoader using the file system."""

    def path_stats(self, path):
        """Return the metadata for the path."""
        st = _path_stat(path)
        return {'mtime': st.st_mtime, 'size': st.st_size}

    def _cache_bytecode(self, source_path, bytecode_path, data):
        mode = _calc_mode(source_path)
        return self.set_data(bytecode_path, data, _mode=mode)

    def set_data(self, path, data, *, _mode=438):
        """Write bytes data to a file."""
        parent, filename = _path_split(path)
        path_parts = []
        while parent and (not _path_isdir(parent)):
            parent, part = _path_split(parent)
            path_parts.append(part)
        for part in reversed(path_parts):
            parent = _path_join(parent, part)
            try:
                _os.mkdir(parent)
            except FileExistsError:
                continue
            except OSError as exc:
                _bootstrap._verbose_message('could not create {!r}: {!r}', parent, exc)
                return
        try:
            _write_atomic(path, data, _mode)
            _bootstrap._verbose_message('created {!r}', path)
        except OSError as exc:
            _bootstrap._verbose_message('could not create {!r}: {!r}', path, exc)