import pickle
import re
import sys
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
class FindDefinitionTest(test_util.TestCase):
    """Test finding definitions relative to various definitions and modules."""

    def setUp(self):
        """Set up module-space.  Starts off empty."""
        self.modules = {}

    def DefineModule(self, name):
        """Define a module and its parents in module space.

        Modules that are already defined in self.modules are not re-created.

        Args:
          name: Fully qualified name of modules to create.

        Returns:
          Deepest nested module.  For example:

            DefineModule('a.b.c')  # Returns c.
        """
        name_path = name.split('.')
        full_path = []
        for node in name_path:
            full_path.append(node)
            full_name = '.'.join(full_path)
            self.modules.setdefault(full_name, types.ModuleType(full_name))
        return self.modules[name]

    def DefineMessage(self, module, name, children=None, add_to_module=True):
        """Define a new Message class in the context of a module.

        Used for easily describing complex Message hierarchy. Message
        is defined including all child definitions.

        Args:
          module: Fully qualified name of module to place Message class in.
          name: Name of Message to define within module.
          children: Define any level of nesting of children
            definitions. To define a message, map the name to another
            dictionary. The dictionary can itself contain additional
            definitions, and so on. To map to an Enum, define the Enum
            class separately and map it by name.
          add_to_module: If True, new Message class is added to
            module. If False, new Message is not added.

        """
        children = children or {}
        module_instance = self.DefineModule(module)
        for attribute, value in children.items():
            if isinstance(value, dict):
                children[attribute] = self.DefineMessage(module, attribute, value, False)
        children['__module__'] = module
        message_class = type(name, (messages.Message,), dict(children))
        if add_to_module:
            setattr(module_instance, name, message_class)
        return message_class

    def Importer(self, module, globals='', locals='', fromlist=None):
        """Importer function.

        Acts like __import__. Only loads modules from self.modules.
        Does not try to load real modules defined elsewhere. Does not
        try to handle relative imports.

        Args:
          module: Fully qualified name of module to load from self.modules.

        """
        if fromlist is None:
            module = module.split('.')[0]
        try:
            return self.modules[module]
        except KeyError:
            raise ImportError()

    def testNoSuchModule(self):
        """Test searching for definitions that do no exist."""
        self.assertRaises(messages.DefinitionNotFoundError, messages.find_definition, 'does.not.exist', importer=self.Importer)

    def testRefersToModule(self):
        """Test that referring to a module does not return that module."""
        self.DefineModule('i.am.a.module')
        self.assertRaises(messages.DefinitionNotFoundError, messages.find_definition, 'i.am.a.module', importer=self.Importer)

    def testNoDefinition(self):
        """Test not finding a definition in an existing module."""
        self.DefineModule('i.am.a.module')
        self.assertRaises(messages.DefinitionNotFoundError, messages.find_definition, 'i.am.a.module.MyMessage', importer=self.Importer)

    def testNotADefinition(self):
        """Test trying to fetch something that is not a definition."""
        module = self.DefineModule('i.am.a.module')
        setattr(module, 'A', 'a string')
        self.assertRaises(messages.DefinitionNotFoundError, messages.find_definition, 'i.am.a.module.A', importer=self.Importer)

    def testGlobalFind(self):
        """Test finding definitions from fully qualified module names."""
        A = self.DefineMessage('a.b.c', 'A', {})
        self.assertEquals(A, messages.find_definition('a.b.c.A', importer=self.Importer))
        B = self.DefineMessage('a.b.c', 'B', {'C': {}})
        self.assertEquals(B.C, messages.find_definition('a.b.c.B.C', importer=self.Importer))

    def testRelativeToModule(self):
        """Test finding definitions relative to modules."""
        a = self.DefineModule('a')
        b = self.DefineModule('a.b')
        c = self.DefineModule('a.b.c')
        A = self.DefineMessage('a', 'A')
        B = self.DefineMessage('a.b', 'B')
        C = self.DefineMessage('a.b.c', 'C')
        D = self.DefineMessage('a.b.d', 'D')
        self.assertEquals(A, messages.find_definition('A', a, importer=self.Importer))
        self.assertEquals(B, messages.find_definition('b.B', a, importer=self.Importer))
        self.assertEquals(C, messages.find_definition('b.c.C', a, importer=self.Importer))
        self.assertEquals(D, messages.find_definition('b.d.D', a, importer=self.Importer))
        self.assertEquals(A, messages.find_definition('A', b, importer=self.Importer))
        self.assertEquals(B, messages.find_definition('B', b, importer=self.Importer))
        self.assertEquals(C, messages.find_definition('c.C', b, importer=self.Importer))
        self.assertEquals(D, messages.find_definition('d.D', b, importer=self.Importer))
        self.assertEquals(A, messages.find_definition('A', c, importer=self.Importer))
        self.assertEquals(B, messages.find_definition('B', c, importer=self.Importer))
        self.assertEquals(C, messages.find_definition('C', c, importer=self.Importer))
        self.assertEquals(D, messages.find_definition('d.D', c, importer=self.Importer))

    def testRelativeToMessages(self):
        """Test finding definitions relative to Message definitions."""
        A = self.DefineMessage('a.b', 'A', {'B': {'C': {}, 'D': {}}})
        B = A.B
        C = A.B.C
        D = A.B.D
        self.assertEquals(A, messages.find_definition('A', A, importer=self.Importer))
        self.assertEquals(B, messages.find_definition('B', A, importer=self.Importer))
        self.assertEquals(C, messages.find_definition('B.C', A, importer=self.Importer))
        self.assertEquals(D, messages.find_definition('B.D', A, importer=self.Importer))
        self.assertEquals(A, messages.find_definition('A', B, importer=self.Importer))
        self.assertEquals(B, messages.find_definition('B', B, importer=self.Importer))
        self.assertEquals(C, messages.find_definition('C', B, importer=self.Importer))
        self.assertEquals(D, messages.find_definition('D', B, importer=self.Importer))
        self.assertEquals(A, messages.find_definition('A', C, importer=self.Importer))
        self.assertEquals(B, messages.find_definition('B', C, importer=self.Importer))
        self.assertEquals(C, messages.find_definition('C', C, importer=self.Importer))
        self.assertEquals(D, messages.find_definition('D', C, importer=self.Importer))
        self.assertEquals(A, messages.find_definition('b.A', C, importer=self.Importer))
        self.assertEquals(B, messages.find_definition('b.A.B', C, importer=self.Importer))
        self.assertEquals(C, messages.find_definition('b.A.B.C', C, importer=self.Importer))
        self.assertEquals(D, messages.find_definition('b.A.B.D', C, importer=self.Importer))

    def testAbsoluteReference(self):
        """Test finding absolute definition names."""
        a = self.DefineModule('a')
        b = self.DefineModule('a.a')
        aA = self.DefineMessage('a', 'A')
        aaA = self.DefineMessage('a.a', 'A')
        self.assertEquals(aA, messages.find_definition('.a.A', None, importer=self.Importer))
        self.assertEquals(aA, messages.find_definition('.a.A', a, importer=self.Importer))
        self.assertEquals(aA, messages.find_definition('.a.A', aA, importer=self.Importer))
        self.assertEquals(aA, messages.find_definition('.a.A', aaA, importer=self.Importer))

    def testFindEnum(self):
        """Test that Enums are found."""

        class Color(messages.Enum):
            pass
        A = self.DefineMessage('a', 'A', {'Color': Color})
        self.assertEquals(Color, messages.find_definition('Color', A, importer=self.Importer))

    def testFalseScope(self):
        """Test Message definitions nested in strange objects are hidden."""
        global X

        class X(object):

            class A(messages.Message):
                pass
        self.assertRaises(TypeError, messages.find_definition, 'A', X)
        self.assertRaises(messages.DefinitionNotFoundError, messages.find_definition, 'X.A', sys.modules[__name__])

    def testSearchAttributeFirst(self):
        """Make sure not faked out by module, but continues searching."""
        A = self.DefineMessage('a', 'A')
        module_A = self.DefineModule('a.A')
        self.assertEquals(A, messages.find_definition('a.A', None, importer=self.Importer))