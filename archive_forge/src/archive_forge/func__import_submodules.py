import argparse
import collections
import importlib
import os
import sys
from tensorflow.python.tools.api.generator import doc_srcs
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export
import sys as _sys
from tensorflow.python.util import module_wrapper as _module_wrapper
def _import_submodules(self):
    """Add imports for all destination modules in self._module_imports."""
    imported_modules = set(self._module_imports.keys())
    for module in imported_modules:
        if not module:
            continue
        module_split = module.split('.')
        parent_module = ''
        for submodule_index in range(len(module_split)):
            if submodule_index > 0:
                submodule = module_split[submodule_index - 1]
                parent_module += '.' + submodule if parent_module else submodule
            import_from = self._output_package
            if self._lazy_loading:
                import_from += '.' + '.'.join(module_split[:submodule_index + 1])
                self.add_import(symbol=None, source_module_name='', source_name=import_from, dest_module_name=parent_module, dest_name=module_split[submodule_index])
            else:
                if self._use_relative_imports:
                    import_from = '.'
                elif submodule_index > 0:
                    import_from += '.' + '.'.join(module_split[:submodule_index])
                self.add_import(symbol=None, source_module_name=import_from, source_name=module_split[submodule_index], dest_module_name=parent_module, dest_name=module_split[submodule_index])