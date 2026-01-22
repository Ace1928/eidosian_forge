import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def _PrintObjects(self, file):
    if self._should_print_single_line:
        self._XCPrint(file, 0, 'objects = {')
    else:
        self._XCPrint(file, 1, 'objects = {\n')
    objects_by_class = {}
    for object in self.Descendants():
        if object == self:
            continue
        class_name = object.__class__.__name__
        if class_name not in objects_by_class:
            objects_by_class[class_name] = []
        objects_by_class[class_name].append(object)
    for class_name in sorted(objects_by_class):
        self._XCPrint(file, 0, '\n')
        self._XCPrint(file, 0, '/* Begin ' + class_name + ' section */\n')
        for object in sorted(objects_by_class[class_name], key=attrgetter('id')):
            object.Print(file)
        self._XCPrint(file, 0, '/* End ' + class_name + ' section */\n')
    if self._should_print_single_line:
        self._XCPrint(file, 0, '}; ')
    else:
        self._XCPrint(file, 1, '};\n')