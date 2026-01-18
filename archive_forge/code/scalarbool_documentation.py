from __future__ import print_function, absolute_import, division, unicode_literals
from ruamel.yaml.anchor import Anchor

You cannot subclass bool, and this is necessary for round-tripping anchored
bool values (and also if you want to preserve the original way of writing)

bool.__bases__ is type 'int', so that is what is used as the basis for ScalarBoolean as well.

You can use these in an if statement, but not when testing equivalence
