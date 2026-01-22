import _imp
import _io
import sys
import _warnings
import marshal
def _get_spec(self, loader_class, fullname, path, smsl, target):
    loader = loader_class(fullname, path)
    return spec_from_file_location(fullname, path, loader=loader, submodule_search_locations=smsl)