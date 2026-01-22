import json
import posixpath
from io import BytesIO, StringIO
from time import time
from unittest import skipIf
from dulwich.tests import TestCase
from ..objects import Blob, Commit, Tag, Tree, parse_timezone
from ..tests.test_object_store import ObjectStoreTests
@skipIf(missing_libs, skipmsg)
class FakeSwiftConnector:

    def __init__(self, root, conf, store=None) -> None:
        if store:
            self.store = store
        else:
            self.store = {}
        self.conf = conf
        self.root = root
        self.concurrency = 1
        self.chunk_length = 12228
        self.cache_length = 1

    def put_object(self, name, content):
        name = posixpath.join(self.root, name)
        if hasattr(content, 'seek'):
            content.seek(0)
            content = content.read()
        self.store[name] = content

    def get_object(self, name, range=None):
        name = posixpath.join(self.root, name)
        if not range:
            try:
                return BytesIO(self.store[name])
            except KeyError:
                return None
        else:
            l, r = range.split('-')
            try:
                if not l:
                    r = -int(r)
                    return self.store[name][r:]
                else:
                    return self.store[name][int(l):int(r)]
            except KeyError:
                return None

    def get_container_objects(self):
        return [{'name': k.replace(self.root + '/', '')} for k in self.store]

    def create_root(self):
        if self.root in self.store.keys():
            pass
        else:
            self.store[self.root] = ''

    def get_object_stat(self, name):
        name = posixpath.join(self.root, name)
        if name not in self.store:
            return None
        return {'content-length': len(self.store[name])}