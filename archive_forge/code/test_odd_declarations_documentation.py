import unittest
from zope.interface import Interface
from zope.interface import classImplements
from zope.interface import classImplementsOnly
from zope.interface import directlyProvidedBy
from zope.interface import directlyProvides
from zope.interface import implementedBy
from zope.interface import implementer
from zope.interface import providedBy
from zope.interface.tests import odd
Test interface declarations against ExtensionClass-like classes.

These tests are to make sure we do something sane in the presence of
classic ExtensionClass classes and instances.
