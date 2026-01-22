from __future__ import unicode_literals
import bisect
import io
import logging
import os
import pkgutil
import sys
import types
import zipimport
from . import DistlibException
from .util import cached_property, get_cache_base, Cache
class ResourceFinder(object):
    """
    Resource finder for file system resources.
    """
    if sys.platform.startswith('java'):
        skipped_extensions = ('.pyc', '.pyo', '.class')
    else:
        skipped_extensions = ('.pyc', '.pyo')

    def __init__(self, module):
        self.module = module
        self.loader = getattr(module, '__loader__', None)
        self.base = os.path.dirname(getattr(module, '__file__', ''))

    def _adjust_path(self, path):
        return os.path.realpath(path)

    def _make_path(self, resource_name):
        if isinstance(resource_name, bytes):
            sep = b'/'
        else:
            sep = '/'
        parts = resource_name.split(sep)
        parts.insert(0, self.base)
        result = os.path.join(*parts)
        return self._adjust_path(result)

    def _find(self, path):
        return os.path.exists(path)

    def get_cache_info(self, resource):
        return (None, resource.path)

    def find(self, resource_name):
        path = self._make_path(resource_name)
        if not self._find(path):
            result = None
        else:
            if self._is_directory(path):
                result = ResourceContainer(self, resource_name)
            else:
                result = Resource(self, resource_name)
            result.path = path
        return result

    def get_stream(self, resource):
        return open(resource.path, 'rb')

    def get_bytes(self, resource):
        with open(resource.path, 'rb') as f:
            return f.read()

    def get_size(self, resource):
        return os.path.getsize(resource.path)

    def get_resources(self, resource):

        def allowed(f):
            return f != '__pycache__' and (not f.endswith(self.skipped_extensions))
        return set([f for f in os.listdir(resource.path) if allowed(f)])

    def is_container(self, resource):
        return self._is_directory(resource.path)
    _is_directory = staticmethod(os.path.isdir)

    def iterator(self, resource_name):
        resource = self.find(resource_name)
        if resource is not None:
            todo = [resource]
            while todo:
                resource = todo.pop(0)
                yield resource
                if resource.is_container:
                    rname = resource.name
                    for name in resource.resources:
                        if not rname:
                            new_name = name
                        else:
                            new_name = '/'.join([rname, name])
                        child = self.find(new_name)
                        if child.is_container:
                            todo.append(child)
                        else:
                            yield child