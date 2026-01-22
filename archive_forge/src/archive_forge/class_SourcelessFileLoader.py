import _imp
import _io
import sys
import _warnings
import marshal
class SourcelessFileLoader(FileLoader, _LoaderBasics):
    """Loader which handles sourceless file imports."""

    def get_code(self, fullname):
        path = self.get_filename(fullname)
        data = self.get_data(path)
        exc_details = {'name': fullname, 'path': path}
        _classify_pyc(data, fullname, exc_details)
        return _compile_bytecode(memoryview(data)[16:], name=fullname, bytecode_path=path)

    def get_source(self, fullname):
        """Return None as there is no source code."""
        return None