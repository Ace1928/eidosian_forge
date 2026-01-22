import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class InterfaceTests(unittest.TestCase):

    def test_attributes_link_to_interface(self):
        from zope.interface import Attribute
        from zope.interface import Interface

        class I1(Interface):
            attr = Attribute('My attr')
        self.assertTrue(I1['attr'].interface is I1)

    def test_methods_link_to_interface(self):
        from zope.interface import Interface

        class I1(Interface):

            def method(foo, bar, bingo):
                """A method"""
        self.assertTrue(I1['method'].interface is I1)

    def test_classImplements_simple(self):
        from zope.interface import Interface
        from zope.interface import implementedBy
        from zope.interface import providedBy

        class ICurrent(Interface):

            def method1(a, b):
                """docstring"""

            def method2(a, b):
                """docstring"""

        class IOther(Interface):
            pass

        class Current:
            __implemented__ = ICurrent

            def method1(self, a, b):
                raise NotImplementedError()

            def method2(self, a, b):
                raise NotImplementedError()
        current = Current()
        self.assertTrue(ICurrent.implementedBy(Current))
        self.assertFalse(IOther.implementedBy(Current))
        self.assertEqual(ICurrent, ICurrent)
        self.assertTrue(ICurrent in implementedBy(Current))
        self.assertFalse(IOther in implementedBy(Current))
        self.assertTrue(ICurrent in providedBy(current))
        self.assertFalse(IOther in providedBy(current))

    def test_classImplements_base_not_derived(self):
        from zope.interface import Interface
        from zope.interface import implementedBy
        from zope.interface import providedBy

        class IBase(Interface):

            def method():
                """docstring"""

        class IDerived(IBase):
            pass

        class Current:
            __implemented__ = IBase

            def method(self):
                raise NotImplementedError()
        current = Current()
        self.assertTrue(IBase.implementedBy(Current))
        self.assertFalse(IDerived.implementedBy(Current))
        self.assertTrue(IBase in implementedBy(Current))
        self.assertFalse(IDerived in implementedBy(Current))
        self.assertTrue(IBase in providedBy(current))
        self.assertFalse(IDerived in providedBy(current))

    def test_classImplements_base_and_derived(self):
        from zope.interface import Interface
        from zope.interface import implementedBy
        from zope.interface import providedBy

        class IBase(Interface):

            def method():
                """docstring"""

        class IDerived(IBase):
            pass

        class Current:
            __implemented__ = IDerived

            def method(self):
                raise NotImplementedError()
        current = Current()
        self.assertTrue(IBase.implementedBy(Current))
        self.assertTrue(IDerived.implementedBy(Current))
        self.assertFalse(IBase in implementedBy(Current))
        self.assertTrue(IBase in implementedBy(Current).flattened())
        self.assertTrue(IDerived in implementedBy(Current))
        self.assertFalse(IBase in providedBy(current))
        self.assertTrue(IBase in providedBy(current).flattened())
        self.assertTrue(IDerived in providedBy(current))

    def test_classImplements_multiple(self):
        from zope.interface import Interface
        from zope.interface import implementedBy
        from zope.interface import providedBy

        class ILeft(Interface):

            def method():
                """docstring"""

        class IRight(ILeft):
            pass

        class Left:
            __implemented__ = ILeft

            def method(self):
                raise NotImplementedError()

        class Right:
            __implemented__ = IRight

        class Ambi(Left, Right):
            pass
        ambi = Ambi()
        self.assertTrue(ILeft.implementedBy(Ambi))
        self.assertTrue(IRight.implementedBy(Ambi))
        self.assertTrue(ILeft in implementedBy(Ambi))
        self.assertTrue(IRight in implementedBy(Ambi))
        self.assertTrue(ILeft in providedBy(ambi))
        self.assertTrue(IRight in providedBy(ambi))

    def test_classImplements_multiple_w_explict_implements(self):
        from zope.interface import Interface
        from zope.interface import implementedBy
        from zope.interface import providedBy

        class ILeft(Interface):

            def method():
                """docstring"""

        class IRight(ILeft):
            pass

        class IOther(Interface):
            pass

        class Left:
            __implemented__ = ILeft

            def method(self):
                raise NotImplementedError()

        class Right:
            __implemented__ = IRight

        class Other:
            __implemented__ = IOther

        class Mixed(Left, Right):
            __implemented__ = (Left.__implemented__, Other.__implemented__)
        mixed = Mixed()
        self.assertTrue(ILeft.implementedBy(Mixed))
        self.assertFalse(IRight.implementedBy(Mixed))
        self.assertTrue(IOther.implementedBy(Mixed))
        self.assertTrue(ILeft in implementedBy(Mixed))
        self.assertFalse(IRight in implementedBy(Mixed))
        self.assertTrue(IOther in implementedBy(Mixed))
        self.assertTrue(ILeft in providedBy(mixed))
        self.assertFalse(IRight in providedBy(mixed))
        self.assertTrue(IOther in providedBy(mixed))

    def testInterfaceExtendsInterface(self):
        from zope.interface import Interface
        new = Interface.__class__
        FunInterface = new('FunInterface')
        BarInterface = new('BarInterface', (FunInterface,))
        BobInterface = new('BobInterface')
        BazInterface = new('BazInterface', (BobInterface, BarInterface))
        self.assertTrue(BazInterface.extends(BobInterface))
        self.assertTrue(BazInterface.extends(BarInterface))
        self.assertTrue(BazInterface.extends(FunInterface))
        self.assertFalse(BobInterface.extends(FunInterface))
        self.assertFalse(BobInterface.extends(BarInterface))
        self.assertTrue(BarInterface.extends(FunInterface))
        self.assertFalse(BarInterface.extends(BazInterface))

    def test_verifyClass(self):
        from zope.interface import Attribute
        from zope.interface import Interface
        from zope.interface.verify import verifyClass

        class ICheckMe(Interface):
            attr = Attribute('My attr')

            def method():
                """A method"""

        class CheckMe:
            __implemented__ = ICheckMe
            attr = 'value'

            def method(self):
                raise NotImplementedError()
        self.assertTrue(verifyClass(ICheckMe, CheckMe))

    def test_verifyObject(self):
        from zope.interface import Attribute
        from zope.interface import Interface
        from zope.interface.verify import verifyObject

        class ICheckMe(Interface):
            attr = Attribute('My attr')

            def method():
                """A method"""

        class CheckMe:
            __implemented__ = ICheckMe
            attr = 'value'

            def method(self):
                raise NotImplementedError()
        check_me = CheckMe()
        self.assertTrue(verifyObject(ICheckMe, check_me))

    def test_interface_object_provides_Interface(self):
        from zope.interface import Interface

        class AnInterface(Interface):
            pass
        self.assertTrue(Interface.providedBy(AnInterface))

    def test_names_simple(self):
        from zope.interface import Attribute
        from zope.interface import Interface

        class ISimple(Interface):
            attr = Attribute('My attr')

            def method():
                """docstring"""
        self.assertEqual(sorted(ISimple.names()), ['attr', 'method'])

    def test_names_derived(self):
        from zope.interface import Attribute
        from zope.interface import Interface

        class IBase(Interface):
            attr = Attribute('My attr')

            def method():
                """docstring"""

        class IDerived(IBase):
            attr2 = Attribute('My attr2')

            def method():
                """docstring"""

            def method2():
                """docstring"""
        self.assertEqual(sorted(IDerived.names()), ['attr2', 'method', 'method2'])
        self.assertEqual(sorted(IDerived.names(all=True)), ['attr', 'attr2', 'method', 'method2'])

    def test_namesAndDescriptions_simple(self):
        from zope.interface import Attribute
        from zope.interface import Interface
        from zope.interface.interface import Method

        class ISimple(Interface):
            attr = Attribute('My attr')

            def method():
                """My method"""
        name_values = sorted(ISimple.namesAndDescriptions())
        self.assertEqual(len(name_values), 2)
        self.assertEqual(name_values[0][0], 'attr')
        self.assertTrue(isinstance(name_values[0][1], Attribute))
        self.assertEqual(name_values[0][1].__name__, 'attr')
        self.assertEqual(name_values[0][1].__doc__, 'My attr')
        self.assertEqual(name_values[1][0], 'method')
        self.assertTrue(isinstance(name_values[1][1], Method))
        self.assertEqual(name_values[1][1].__name__, 'method')
        self.assertEqual(name_values[1][1].__doc__, 'My method')

    def test_namesAndDescriptions_derived(self):
        from zope.interface import Attribute
        from zope.interface import Interface
        from zope.interface.interface import Method

        class IBase(Interface):
            attr = Attribute('My attr')

            def method():
                """My method"""

        class IDerived(IBase):
            attr2 = Attribute('My attr2')

            def method():
                """My method, overridden"""

            def method2():
                """My method2"""
        name_values = sorted(IDerived.namesAndDescriptions())
        self.assertEqual(len(name_values), 3)
        self.assertEqual(name_values[0][0], 'attr2')
        self.assertTrue(isinstance(name_values[0][1], Attribute))
        self.assertEqual(name_values[0][1].__name__, 'attr2')
        self.assertEqual(name_values[0][1].__doc__, 'My attr2')
        self.assertEqual(name_values[1][0], 'method')
        self.assertTrue(isinstance(name_values[1][1], Method))
        self.assertEqual(name_values[1][1].__name__, 'method')
        self.assertEqual(name_values[1][1].__doc__, 'My method, overridden')
        self.assertEqual(name_values[2][0], 'method2')
        self.assertTrue(isinstance(name_values[2][1], Method))
        self.assertEqual(name_values[2][1].__name__, 'method2')
        self.assertEqual(name_values[2][1].__doc__, 'My method2')
        name_values = sorted(IDerived.namesAndDescriptions(all=True))
        self.assertEqual(len(name_values), 4)
        self.assertEqual(name_values[0][0], 'attr')
        self.assertTrue(isinstance(name_values[0][1], Attribute))
        self.assertEqual(name_values[0][1].__name__, 'attr')
        self.assertEqual(name_values[0][1].__doc__, 'My attr')
        self.assertEqual(name_values[1][0], 'attr2')
        self.assertTrue(isinstance(name_values[1][1], Attribute))
        self.assertEqual(name_values[1][1].__name__, 'attr2')
        self.assertEqual(name_values[1][1].__doc__, 'My attr2')
        self.assertEqual(name_values[2][0], 'method')
        self.assertTrue(isinstance(name_values[2][1], Method))
        self.assertEqual(name_values[2][1].__name__, 'method')
        self.assertEqual(name_values[2][1].__doc__, 'My method, overridden')
        self.assertEqual(name_values[3][0], 'method2')
        self.assertTrue(isinstance(name_values[3][1], Method))
        self.assertEqual(name_values[3][1].__name__, 'method2')
        self.assertEqual(name_values[3][1].__doc__, 'My method2')

    def test_getDescriptionFor_nonesuch_no_default(self):
        from zope.interface import Interface

        class IEmpty(Interface):
            pass
        self.assertRaises(KeyError, IEmpty.getDescriptionFor, 'nonesuch')

    def test_getDescriptionFor_simple(self):
        from zope.interface import Attribute
        from zope.interface import Interface
        from zope.interface.interface import Method

        class ISimple(Interface):
            attr = Attribute('My attr')

            def method():
                """My method"""
        a_desc = ISimple.getDescriptionFor('attr')
        self.assertTrue(isinstance(a_desc, Attribute))
        self.assertEqual(a_desc.__name__, 'attr')
        self.assertEqual(a_desc.__doc__, 'My attr')
        m_desc = ISimple.getDescriptionFor('method')
        self.assertTrue(isinstance(m_desc, Method))
        self.assertEqual(m_desc.__name__, 'method')
        self.assertEqual(m_desc.__doc__, 'My method')

    def test_getDescriptionFor_derived(self):
        from zope.interface import Attribute
        from zope.interface import Interface
        from zope.interface.interface import Method

        class IBase(Interface):
            attr = Attribute('My attr')

            def method():
                """My method"""

        class IDerived(IBase):
            attr2 = Attribute('My attr2')

            def method():
                """My method, overridden"""

            def method2():
                """My method2"""
        a_desc = IDerived.getDescriptionFor('attr')
        self.assertTrue(isinstance(a_desc, Attribute))
        self.assertEqual(a_desc.__name__, 'attr')
        self.assertEqual(a_desc.__doc__, 'My attr')
        m_desc = IDerived.getDescriptionFor('method')
        self.assertTrue(isinstance(m_desc, Method))
        self.assertEqual(m_desc.__name__, 'method')
        self.assertEqual(m_desc.__doc__, 'My method, overridden')
        a2_desc = IDerived.getDescriptionFor('attr2')
        self.assertTrue(isinstance(a2_desc, Attribute))
        self.assertEqual(a2_desc.__name__, 'attr2')
        self.assertEqual(a2_desc.__doc__, 'My attr2')
        m2_desc = IDerived.getDescriptionFor('method2')
        self.assertTrue(isinstance(m2_desc, Method))
        self.assertEqual(m2_desc.__name__, 'method2')
        self.assertEqual(m2_desc.__doc__, 'My method2')

    def test___getitem__nonesuch(self):
        from zope.interface import Interface

        class IEmpty(Interface):
            pass
        self.assertRaises(KeyError, IEmpty.__getitem__, 'nonesuch')

    def test___getitem__simple(self):
        from zope.interface import Attribute
        from zope.interface import Interface
        from zope.interface.interface import Method

        class ISimple(Interface):
            attr = Attribute('My attr')

            def method():
                """My method"""
        a_desc = ISimple['attr']
        self.assertTrue(isinstance(a_desc, Attribute))
        self.assertEqual(a_desc.__name__, 'attr')
        self.assertEqual(a_desc.__doc__, 'My attr')
        m_desc = ISimple['method']
        self.assertTrue(isinstance(m_desc, Method))
        self.assertEqual(m_desc.__name__, 'method')
        self.assertEqual(m_desc.__doc__, 'My method')

    def test___getitem___derived(self):
        from zope.interface import Attribute
        from zope.interface import Interface
        from zope.interface.interface import Method

        class IBase(Interface):
            attr = Attribute('My attr')

            def method():
                """My method"""

        class IDerived(IBase):
            attr2 = Attribute('My attr2')

            def method():
                """My method, overridden"""

            def method2():
                """My method2"""
        a_desc = IDerived['attr']
        self.assertTrue(isinstance(a_desc, Attribute))
        self.assertEqual(a_desc.__name__, 'attr')
        self.assertEqual(a_desc.__doc__, 'My attr')
        m_desc = IDerived['method']
        self.assertTrue(isinstance(m_desc, Method))
        self.assertEqual(m_desc.__name__, 'method')
        self.assertEqual(m_desc.__doc__, 'My method, overridden')
        a2_desc = IDerived['attr2']
        self.assertTrue(isinstance(a2_desc, Attribute))
        self.assertEqual(a2_desc.__name__, 'attr2')
        self.assertEqual(a2_desc.__doc__, 'My attr2')
        m2_desc = IDerived['method2']
        self.assertTrue(isinstance(m2_desc, Method))
        self.assertEqual(m2_desc.__name__, 'method2')
        self.assertEqual(m2_desc.__doc__, 'My method2')

    def test___contains__nonesuch(self):
        from zope.interface import Interface

        class IEmpty(Interface):
            pass
        self.assertFalse('nonesuch' in IEmpty)

    def test___contains__simple(self):
        from zope.interface import Attribute
        from zope.interface import Interface

        class ISimple(Interface):
            attr = Attribute('My attr')

            def method():
                """My method"""
        self.assertTrue('attr' in ISimple)
        self.assertTrue('method' in ISimple)

    def test___contains__derived(self):
        from zope.interface import Attribute
        from zope.interface import Interface

        class IBase(Interface):
            attr = Attribute('My attr')

            def method():
                """My method"""

        class IDerived(IBase):
            attr2 = Attribute('My attr2')

            def method():
                """My method, overridden"""

            def method2():
                """My method2"""
        self.assertTrue('attr' in IDerived)
        self.assertTrue('method' in IDerived)
        self.assertTrue('attr2' in IDerived)
        self.assertTrue('method2' in IDerived)

    def test___iter__empty(self):
        from zope.interface import Interface

        class IEmpty(Interface):
            pass
        self.assertEqual(list(IEmpty), [])

    def test___iter__simple(self):
        from zope.interface import Attribute
        from zope.interface import Interface

        class ISimple(Interface):
            attr = Attribute('My attr')

            def method():
                """My method"""
        self.assertEqual(sorted(list(ISimple)), ['attr', 'method'])

    def test___iter__derived(self):
        from zope.interface import Attribute
        from zope.interface import Interface

        class IBase(Interface):
            attr = Attribute('My attr')

            def method():
                """My method"""

        class IDerived(IBase):
            attr2 = Attribute('My attr2')

            def method():
                """My method, overridden"""

            def method2():
                """My method2"""
        self.assertEqual(sorted(list(IDerived)), ['attr', 'attr2', 'method', 'method2'])

    def test_function_attributes_become_tagged_values(self):
        from zope.interface import Interface

        class ITagMe(Interface):

            def method():
                """docstring"""
            method.optional = 1
        method = ITagMe['method']
        self.assertEqual(method.getTaggedValue('optional'), 1)

    def test___doc___non_element(self):
        from zope.interface import Interface

        class IHaveADocString(Interface):
            """xxx"""
        self.assertEqual(IHaveADocString.__doc__, 'xxx')
        self.assertEqual(list(IHaveADocString), [])

    def test___doc___as_element(self):
        from zope.interface import Attribute
        from zope.interface import Interface

        class IHaveADocString(Interface):
            """xxx"""
            __doc__ = Attribute('the doc')
        self.assertEqual(IHaveADocString.__doc__, '')
        self.assertEqual(list(IHaveADocString), ['__doc__'])

    def _errorsEqual(self, has_invariant, error_len, error_msgs, iface):
        from zope.interface.exceptions import Invalid
        self.assertRaises(Invalid, iface.validateInvariants, has_invariant)
        e = []
        try:
            iface.validateInvariants(has_invariant, e)
            self.fail('validateInvariants should always raise')
        except Invalid as error:
            self.assertEqual(error.args[0], e)
        self.assertEqual(len(e), error_len)
        msgs = [error.args[0] for error in e]
        msgs.sort()
        for msg in msgs:
            self.assertEqual(msg, error_msgs.pop(0))

    def test_invariant_simple(self):
        from zope.interface import Attribute
        from zope.interface import Interface
        from zope.interface import directlyProvides
        from zope.interface import invariant

        class IInvariant(Interface):
            foo = Attribute('foo')
            bar = Attribute('bar; must eval to Boolean True if foo does')
            invariant(_ifFooThenBar)

        class HasInvariant:
            pass
        has_invariant = HasInvariant()
        directlyProvides(has_invariant, IInvariant)
        self.assertEqual(IInvariant.getTaggedValue('invariants'), [_ifFooThenBar])
        self.assertEqual(IInvariant.validateInvariants(has_invariant), None)
        has_invariant.bar = 27
        self.assertEqual(IInvariant.validateInvariants(has_invariant), None)
        has_invariant.foo = 42
        self.assertEqual(IInvariant.validateInvariants(has_invariant), None)
        del has_invariant.bar
        self._errorsEqual(has_invariant, 1, ['If Foo, then Bar!'], IInvariant)

    def test_invariant_nested(self):
        from zope.interface import Attribute
        from zope.interface import Interface
        from zope.interface import directlyProvides
        from zope.interface import invariant

        class IInvariant(Interface):
            foo = Attribute('foo')
            bar = Attribute('bar; must eval to Boolean True if foo does')
            invariant(_ifFooThenBar)

        class ISubInvariant(IInvariant):
            invariant(_barGreaterThanFoo)

        class HasInvariant:
            pass
        self.assertEqual(ISubInvariant.getTaggedValue('invariants'), [_barGreaterThanFoo])
        has_invariant = HasInvariant()
        directlyProvides(has_invariant, ISubInvariant)
        has_invariant.foo = 42
        self._errorsEqual(has_invariant, 1, ['If Foo, then Bar!'], ISubInvariant)
        has_invariant.foo = 2
        has_invariant.bar = 1
        self._errorsEqual(has_invariant, 1, ['Please, Boo MUST be greater than Foo!'], ISubInvariant)
        has_invariant.foo = 1
        has_invariant.bar = 0
        self._errorsEqual(has_invariant, 2, ['If Foo, then Bar!', 'Please, Boo MUST be greater than Foo!'], ISubInvariant)
        has_invariant.foo = 1
        has_invariant.bar = 2
        self.assertEqual(IInvariant.validateInvariants(has_invariant), None)

    def test_invariant_mutandis(self):
        from zope.interface import Attribute
        from zope.interface import Interface
        from zope.interface import directlyProvides
        from zope.interface import invariant

        class IInvariant(Interface):
            foo = Attribute('foo')
            bar = Attribute('bar; must eval to Boolean True if foo does')
            invariant(_ifFooThenBar)

        class HasInvariant:
            pass
        has_invariant = HasInvariant()
        directlyProvides(has_invariant, IInvariant)
        has_invariant.foo = 42
        old_invariants = IInvariant.getTaggedValue('invariants')
        invariants = old_invariants[:]
        invariants.append(_barGreaterThanFoo)
        IInvariant.setTaggedValue('invariants', invariants)
        self._errorsEqual(has_invariant, 1, ['If Foo, then Bar!'], IInvariant)
        has_invariant.foo = 2
        has_invariant.bar = 1
        self._errorsEqual(has_invariant, 1, ['Please, Boo MUST be greater than Foo!'], IInvariant)
        has_invariant.foo = 1
        has_invariant.bar = 0
        self._errorsEqual(has_invariant, 2, ['If Foo, then Bar!', 'Please, Boo MUST be greater than Foo!'], IInvariant)
        has_invariant.foo = 1
        has_invariant.bar = 2
        self.assertEqual(IInvariant.validateInvariants(has_invariant), None)
        IInvariant.setTaggedValue('invariants', old_invariants)

    def test___doc___element(self):
        from zope.interface import Attribute
        from zope.interface import Interface

        class IDocstring(Interface):
            """xxx"""
        self.assertEqual(IDocstring.__doc__, 'xxx')
        self.assertEqual(list(IDocstring), [])

        class IDocstringAndAttribute(Interface):
            """xxx"""
            __doc__ = Attribute('the doc')
        self.assertEqual(IDocstringAndAttribute.__doc__, '')
        self.assertEqual(list(IDocstringAndAttribute), ['__doc__'])

    def test_invariant_as_decorator(self):
        from zope.interface import Attribute
        from zope.interface import Interface
        from zope.interface import implementer
        from zope.interface import invariant
        from zope.interface.exceptions import Invalid

        class IRange(Interface):
            min = Attribute('Lower bound')
            max = Attribute('Upper bound')

            @invariant
            def range_invariant(ob):
                if ob.max < ob.min:
                    raise Invalid('max < min')

        @implementer(IRange)
        class Range:

            def __init__(self, min, max):
                self.min, self.max = (min, max)
        IRange.validateInvariants(Range(1, 2))
        IRange.validateInvariants(Range(1, 1))
        try:
            IRange.validateInvariants(Range(2, 1))
        except Invalid as e:
            self.assertEqual(str(e), 'max < min')

    def test_taggedValue(self):
        from zope.interface import Attribute
        from zope.interface import Interface
        from zope.interface import taggedValue

        class ITagged(Interface):
            foo = Attribute('foo')
            bar = Attribute('bar; must eval to Boolean True if foo does')
            taggedValue('qux', 'Spam')

        class IDerived(ITagged):
            taggedValue('qux', 'Spam Spam')
            taggedValue('foo', 'bar')

        class IDerived2(IDerived):
            pass
        self.assertEqual(ITagged.getTaggedValue('qux'), 'Spam')
        self.assertRaises(KeyError, ITagged.getTaggedValue, 'foo')
        self.assertEqual(list(ITagged.getTaggedValueTags()), ['qux'])
        self.assertEqual(IDerived2.getTaggedValue('qux'), 'Spam Spam')
        self.assertEqual(IDerived2.getTaggedValue('foo'), 'bar')
        self.assertEqual(set(IDerived2.getTaggedValueTags()), {'qux', 'foo'})

    def _make_taggedValue_tree(self, base):
        from zope.interface import Attribute
        from zope.interface import taggedValue
        O = base

        class F(O):
            taggedValue('tag', 'F')
            tag = Attribute('F')

        class E(O):
            taggedValue('tag', 'E')
            tag = Attribute('E')

        class D(O):
            taggedValue('tag', 'D')
            tag = Attribute('D')

        class C(D, F):
            taggedValue('tag', 'C')
            tag = Attribute('C')

        class B(D, E):
            pass

        class A(B, C):
            pass
        return A

    def test_getTaggedValue_follows__iro__(self):
        from zope.interface import Interface
        class_A = self._make_taggedValue_tree(object)
        self.assertEqual(class_A.tag.__name__, 'C')
        iface_A = self._make_taggedValue_tree(Interface)
        self.assertEqual(iface_A['tag'].__name__, 'C')
        self.assertEqual(iface_A.getTaggedValue('tag'), 'C')
        self.assertEqual(iface_A.queryTaggedValue('tag'), 'C')
        assert iface_A.__bases__[0].__name__ == 'B'
        iface_A.__bases__[0].setTaggedValue('tag', 'B')
        self.assertEqual(iface_A.getTaggedValue('tag'), 'B')

    def test_getDirectTaggedValue_ignores__iro__(self):
        from zope.interface import Interface
        A = self._make_taggedValue_tree(Interface)
        self.assertIsNone(A.queryDirectTaggedValue('tag'))
        self.assertEqual([], list(A.getDirectTaggedValueTags()))
        with self.assertRaises(KeyError):
            A.getDirectTaggedValue('tag')
        A.setTaggedValue('tag', 'A')
        self.assertEqual(A.queryDirectTaggedValue('tag'), 'A')
        self.assertEqual(A.getDirectTaggedValue('tag'), 'A')
        self.assertEqual(['tag'], list(A.getDirectTaggedValueTags()))
        assert A.__bases__[1].__name__ == 'C'
        C = A.__bases__[1]
        self.assertEqual(C.queryDirectTaggedValue('tag'), 'C')
        self.assertEqual(C.getDirectTaggedValue('tag'), 'C')
        self.assertEqual(['tag'], list(C.getDirectTaggedValueTags()))

    def test_description_cache_management(self):
        from zope.interface import Attribute
        from zope.interface import Interface

        class I1(Interface):
            a = Attribute('a')

        class I2(I1):
            pass

        class I3(I2):
            pass
        self.assertTrue(I3.get('a') is I1.get('a'))
        I2.__bases__ = (Interface,)
        self.assertTrue(I3.get('a') is None)

    def test___call___defers_to___conform___(self):
        from zope.interface import Interface
        from zope.interface import implementer

        class I(Interface):
            pass

        @implementer(I)
        class C:

            def __conform__(self, proto):
                return 0
        self.assertEqual(I(C()), 0)

    def test___call___object_implements(self):
        from zope.interface import Interface
        from zope.interface import implementer

        class I(Interface):
            pass

        @implementer(I)
        class C:
            pass
        c = C()
        self.assertTrue(I(c) is c)

    def test___call___miss_wo_alternate(self):
        from zope.interface import Interface

        class I(Interface):
            pass

        class C:
            pass
        c = C()
        self.assertRaises(TypeError, I, c)

    def test___call___miss_w_alternate(self):
        from zope.interface import Interface

        class I(Interface):
            pass

        class C:
            pass
        c = C()
        self.assertTrue(I(c, self) is self)

    def test___call___w_adapter_hook(self):
        from zope.interface import Interface
        from zope.interface.interface import adapter_hooks

        def _miss(iface, obj):
            pass

        def _hit(iface, obj):
            return self

        class I(Interface):
            pass

        class C:
            pass
        c = C()
        old_adapter_hooks = adapter_hooks[:]
        adapter_hooks[:] = [_miss, _hit]
        try:
            self.assertTrue(I(c) is self)
        finally:
            adapter_hooks[:] = old_adapter_hooks

    def test___call___w_overridden_adapt(self):
        from zope.interface import Interface
        from zope.interface import implementer
        from zope.interface import interfacemethod

        class I(Interface):

            @interfacemethod
            def __adapt__(self, obj):
                return 42

        @implementer(I)
        class O:
            pass
        self.assertEqual(42, I(object()))
        self.assertEqual(42, I(O()))

    def test___call___w_overridden_adapt_and_conform(self):
        from zope.interface import Interface
        from zope.interface import implementer
        from zope.interface import interfacemethod

        class IAdapt(Interface):

            @interfacemethod
            def __adapt__(self, obj):
                return 42

        class ISimple(Interface):
            """Nothing special."""

        @implementer(IAdapt)
        class Conform24:

            def __conform__(self, iface):
                return 24

        @implementer(IAdapt)
        class ConformNone:

            def __conform__(self, iface):
                return None
        self.assertEqual(42, IAdapt(object()))
        self.assertEqual(24, ISimple(Conform24()))
        self.assertEqual(24, IAdapt(Conform24()))
        with self.assertRaises(TypeError):
            ISimple(ConformNone())
        self.assertEqual(42, IAdapt(ConformNone()))

    def test___call___w_overridden_adapt_call_super(self):
        import sys
        from zope.interface import Interface
        from zope.interface import implementer
        from zope.interface import interfacemethod

        class I(Interface):

            @interfacemethod
            def __adapt__(self, obj):
                if not self.providedBy(obj):
                    return 42
                return super().__adapt__(obj)

        @implementer(I)
        class O:
            pass
        self.assertEqual(42, I(object()))
        o = O()
        self.assertIs(o, I(o))

    def test___adapt___as_method_and_implementation(self):
        from zope.interface import Interface
        from zope.interface import interfacemethod

        class I(Interface):

            @interfacemethod
            def __adapt__(self, obj):
                return 42

            def __adapt__(to_adapt):
                """This is a protocol"""
        self.assertEqual(42, I(object()))
        self.assertEqual(I['__adapt__'].getSignatureString(), '(to_adapt)')

    def test___adapt__inheritance_and_type(self):
        from zope.interface import Interface
        from zope.interface import interfacemethod

        class IRoot(Interface):
            """Root"""

        class IWithAdapt(IRoot):

            @interfacemethod
            def __adapt__(self, obj):
                return 42

        class IOther(IRoot):
            """Second branch"""

        class IUnrelated(Interface):
            """Unrelated"""

        class IDerivedAdapt(IUnrelated, IWithAdapt, IOther):
            """Inherits an adapt"""

        class IDerived2Adapt(IDerivedAdapt):
            """Overrides an inherited custom adapt."""

            @interfacemethod
            def __adapt__(self, obj):
                return 24
        self.assertEqual(42, IDerivedAdapt(object()))
        for iface in (IRoot, IWithAdapt, IOther, IUnrelated, IDerivedAdapt):
            self.assertEqual(__name__, iface.__module__)
        for iface in (IRoot, IOther, IUnrelated):
            self.assertEqual(type(IRoot), type(Interface))
        self.assertNotEqual(type(Interface), type(IWithAdapt))
        self.assertEqual(type(IWithAdapt), type(IDerivedAdapt))
        self.assertIsInstance(IWithAdapt, type(Interface))
        self.assertEqual(24, IDerived2Adapt(object()))
        self.assertNotEqual(type(IDerived2Adapt), type(IDerivedAdapt))
        self.assertIsInstance(IDerived2Adapt, type(IDerivedAdapt))

    def test_interfacemethod_is_general(self):
        from zope.interface import Interface
        from zope.interface import interfacemethod

        class I(Interface):

            @interfacemethod
            def __call__(self, obj):
                """Replace an existing method"""
                return 42

            @interfacemethod
            def this_is_new(self):
                return 42
        self.assertEqual(I(self), 42)
        self.assertEqual(I.this_is_new(), 42)