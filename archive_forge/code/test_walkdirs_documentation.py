import os
from breezy import tests
from breezy.mutabletree import MutableTree
from breezy.osutils import supports_symlinks
from breezy.tests.per_tree import TestCaseWithTree
Tests for the generic Tree.walkdirs interface.