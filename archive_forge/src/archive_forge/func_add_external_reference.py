from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import collections
import six
def add_external_reference(self, name, node, name_ref=None):
    ref = ExternalReference(name=name, node=node, name_ref=name_ref)
    if name in self.external_references:
        self.external_references[name].append(ref)
    else:
        self.external_references[name] = [ref]