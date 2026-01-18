import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_read_shelf(self):
    manager = self.get_manager()
    shelf_id, shelf_file = manager.new_shelf()
    try:
        shelf_file.write(b'foo')
    finally:
        shelf_file.close()
    shelf_id, shelf_file = manager.new_shelf()
    try:
        shelf_file.write(b'bar')
    finally:
        shelf_file.close()
    shelf_file = manager.read_shelf(1)
    try:
        self.assertEqual(b'foo', shelf_file.read())
    finally:
        shelf_file.close()
    shelf_file = manager.read_shelf(2)
    try:
        self.assertEqual(b'bar', shelf_file.read())
    finally:
        shelf_file.close()