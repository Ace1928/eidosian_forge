import sys
import os
class VerifyShadowedImport(object):

    def __init__(self, import_name):
        self.import_name = import_name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            if exc_type == DebuggerInitializationError:
                return False
            found_at = find_in_pythonpath(self.import_name)
            if len(found_at) <= 1:
                return False
            msg = self._generate_shadowed_import_message(found_at)
            raise DebuggerInitializationError(msg)

    def _generate_shadowed_import_message(self, found_at):
        msg = 'It was not possible to initialize the debugger due to a module name conflict.\n\ni.e.: the module "%(import_name)s" could not be imported because it is shadowed by:\n%(found_at)s\nPlease rename this file/folder so that the original module from the standard library can be imported.' % {'import_name': self.import_name, 'found_at': found_at[0]}
        return msg

    def check(self, module, expected_attributes):
        msg = ''
        for expected_attribute in expected_attributes:
            try:
                getattr(module, expected_attribute)
            except:
                msg = self._generate_shadowed_import_message([module.__file__])
                break
        if msg:
            raise DebuggerInitializationError(msg)