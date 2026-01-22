from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
class TestTestResource(testtools.TestCase):

    def testUnimplementedGetResource(self):
        resource_manager = testresources.TestResource()
        self.assertRaises(NotImplementedError, resource_manager.getResource)

    def testInitiallyNotDirty(self):
        resource_manager = testresources.TestResource()
        self.assertEqual(False, resource_manager._dirty)

    def testInitiallyUnused(self):
        resource_manager = testresources.TestResource()
        self.assertEqual(0, resource_manager._uses)

    def testInitiallyNoCurrentResource(self):
        resource_manager = testresources.TestResource()
        self.assertEqual(None, resource_manager._currentResource)

    def testneededResourcesDefault(self):
        resource = testresources.TestResource()
        self.assertEqual([resource], resource.neededResources())

    def testneededResourcesDependenciesFirst(self):
        resource = testresources.TestResource()
        dep1 = testresources.TestResource()
        dep2 = testresources.TestResource()
        resource.resources.append(('dep1', dep1))
        resource.resources.append(('dep2', dep2))
        self.assertEqual([dep1, dep2, resource], resource.neededResources())

    def testneededResourcesClosure(self):
        resource = testresources.TestResource()
        dep1 = testresources.TestResource()
        dep2 = testresources.TestResource()
        resource.resources.append(('dep1', dep1))
        dep1.resources.append(('dep2', dep2))
        self.assertEqual([dep2, dep1, resource], resource.neededResources())

    def testDefaultCosts(self):
        resource_manager = testresources.TestResource()
        self.assertEqual(resource_manager.setUpCost, 1)
        self.assertEqual(resource_manager.tearDownCost, 1)

    def testGetResourceReturnsMakeResource(self):
        resource_manager = MockResource()
        resource = resource_manager.getResource()
        self.assertEqual(resource_manager.make({}), resource)

    def testGetResourceIncrementsUses(self):
        resource_manager = MockResource()
        resource_manager.getResource()
        self.assertEqual(1, resource_manager._uses)
        resource_manager.getResource()
        self.assertEqual(2, resource_manager._uses)

    def testGetResourceDoesntDirty(self):
        resource_manager = MockResource()
        resource_manager.getResource()
        self.assertEqual(resource_manager._dirty, False)

    def testGetResourceSetsCurrentResource(self):
        resource_manager = MockResource()
        resource = resource_manager.getResource()
        self.assertIs(resource_manager._currentResource, resource)

    def testGetResourceTwiceReturnsIdenticalResource(self):
        resource_manager = MockResource()
        resource1 = resource_manager.getResource()
        resource2 = resource_manager.getResource()
        self.assertIs(resource1, resource2)

    def testGetResourceCallsMakeResource(self):
        resource_manager = MockResource()
        resource_manager.getResource()
        self.assertEqual(1, resource_manager.makes)

    def testIsDirty(self):
        resource_manager = MockResource()
        r = resource_manager.getResource()
        resource_manager.dirtied(r)
        self.assertTrue(resource_manager.isDirty())
        resource_manager.finishedWith(r)

    def testIsDirtyIsTrueIfDependenciesChanged(self):
        resource_manager = MockResource()
        dep1 = MockResource()
        dep2 = MockResource()
        dep3 = MockResource()
        resource_manager.resources.append(('dep1', dep1))
        resource_manager.resources.append(('dep2', dep2))
        resource_manager.resources.append(('dep3', dep3))
        r = resource_manager.getResource()
        dep2.dirtied(r.dep2)
        r2 = dep2.getResource()
        self.assertTrue(resource_manager.isDirty())
        resource_manager.finishedWith(r)
        dep2.finishedWith(r2)

    def testIsDirtyIsTrueIfDependenciesAreDirty(self):
        resource_manager = MockResource()
        dep1 = MockResource()
        dep2 = MockResource()
        dep3 = MockResource()
        resource_manager.resources.append(('dep1', dep1))
        resource_manager.resources.append(('dep2', dep2))
        resource_manager.resources.append(('dep3', dep3))
        r = resource_manager.getResource()
        dep2.dirtied(r.dep2)
        self.assertTrue(resource_manager.isDirty())
        resource_manager.finishedWith(r)

    def testRepeatedGetResourceCallsMakeResourceOnceOnly(self):
        resource_manager = MockResource()
        resource_manager.getResource()
        resource_manager.getResource()
        self.assertEqual(1, resource_manager.makes)

    def testGetResourceResetsUsedResource(self):
        resource_manager = MockResettableResource()
        resource_manager.getResource()
        resource = resource_manager.getResource()
        self.assertEqual(1, resource_manager.makes)
        resource_manager.dirtied(resource)
        resource_manager.getResource()
        self.assertEqual(1, resource_manager.makes)
        self.assertEqual(1, resource_manager.resets)
        resource_manager.finishedWith(resource)

    def testIsResetIfDependenciesAreDirty(self):
        resource_manager = MockResource()
        dep1 = MockResettableResource()
        resource_manager.resources.append(('dep1', dep1))
        r = resource_manager.getResource()
        dep1.dirtied(r.dep1)
        r = resource_manager.getResource()
        self.assertFalse(resource_manager.isDirty())
        self.assertFalse(dep1.isDirty())
        resource_manager.finishedWith(r)
        resource_manager.finishedWith(r)

    def testUsedResourceResetBetweenUses(self):
        resource_manager = MockResettableResource()
        resource_manager.getResource()
        resource = resource_manager.getResource()
        resource_manager.dirtied(resource)
        resource_manager.finishedWith(resource)
        resource = resource_manager.getResource()
        resource_manager.finishedWith(resource)
        resource_manager.finishedWith(resource)
        self.assertEqual(1, resource_manager.makes)
        self.assertEqual(1, resource_manager.resets)
        self.assertEqual(1, resource_manager.cleans)

    def testFinishedWithDecrementsUses(self):
        resource_manager = MockResource()
        resource = resource_manager.getResource()
        resource = resource_manager.getResource()
        self.assertEqual(2, resource_manager._uses)
        resource_manager.finishedWith(resource)
        self.assertEqual(1, resource_manager._uses)
        resource_manager.finishedWith(resource)
        self.assertEqual(0, resource_manager._uses)

    def testFinishedWithResetsCurrentResource(self):
        resource_manager = MockResource()
        resource = resource_manager.getResource()
        resource_manager.finishedWith(resource)
        self.assertIs(None, resource_manager._currentResource)

    def testFinishedWithCallsCleanResource(self):
        resource_manager = MockResource()
        resource = resource_manager.getResource()
        resource_manager.finishedWith(resource)
        self.assertEqual(1, resource_manager.cleans)

    def testUsingTwiceMakesAndCleansTwice(self):
        resource_manager = MockResource()
        resource = resource_manager.getResource()
        resource_manager.finishedWith(resource)
        resource = resource_manager.getResource()
        resource_manager.finishedWith(resource)
        self.assertEqual(2, resource_manager.makes)
        self.assertEqual(2, resource_manager.cleans)

    def testFinishedWithCallsCleanResourceOnceOnly(self):
        resource_manager = MockResource()
        resource = resource_manager.getResource()
        resource = resource_manager.getResource()
        resource_manager.finishedWith(resource)
        self.assertEqual(0, resource_manager.cleans)
        resource_manager.finishedWith(resource)
        self.assertEqual(1, resource_manager.cleans)

    def testFinishedWithMarksNonDirty(self):
        resource_manager = MockResource()
        resource = resource_manager.getResource()
        resource_manager.dirtied(resource)
        resource_manager.finishedWith(resource)
        self.assertEqual(False, resource_manager._dirty)

    def testResourceAvailableBetweenFinishedWithCalls(self):
        resource_manager = MockResource()
        resource = resource_manager.getResource()
        resource = resource_manager.getResource()
        resource_manager.finishedWith(resource)
        self.assertIs(resource, resource_manager._currentResource)
        resource_manager.finishedWith(resource)

    def testDirtiedSetsDirty(self):
        resource_manager = MockResource()
        resource = resource_manager.getResource()
        self.assertEqual(False, resource_manager._dirty)
        resource_manager.dirtied(resource)
        self.assertEqual(True, resource_manager._dirty)

    def testDirtyingResourceTriggersCleanOnGet(self):
        resource_manager = MockResource()
        resource1 = resource_manager.getResource()
        resource2 = resource_manager.getResource()
        resource_manager.dirtied(resource2)
        resource_manager.finishedWith(resource2)
        self.assertEqual(0, resource_manager.cleans)
        resource3 = resource_manager.getResource()
        self.assertEqual(1, resource_manager.cleans)
        resource_manager.finishedWith(resource3)
        resource_manager.finishedWith(resource1)
        self.assertEqual(2, resource_manager.cleans)

    def testDefaultResetMethodPreservesCleanResource(self):
        resource_manager = MockResource()
        resource = resource_manager.getResource()
        self.assertEqual(1, resource_manager.makes)
        self.assertEqual(False, resource_manager._dirty)
        resource_manager.reset(resource)
        self.assertEqual(1, resource_manager.makes)
        self.assertEqual(0, resource_manager.cleans)

    def testDefaultResetMethodRecreatesDirtyResource(self):
        resource_manager = MockResource()
        resource = resource_manager.getResource()
        self.assertEqual(1, resource_manager.makes)
        resource_manager.dirtied(resource)
        resource_manager.reset(resource)
        self.assertEqual(2, resource_manager.makes)
        self.assertEqual(1, resource_manager.cleans)

    def testDefaultResetResetsDependencies(self):
        resource_manager = MockResettableResource()
        dep1 = MockResettableResource()
        dep2 = MockResettableResource()
        resource_manager.resources.append(('dep1', dep1))
        resource_manager.resources.append(('dep2', dep2))
        r_outer = resource_manager.getResource()
        r_inner = resource_manager.getResource()
        dep2.dirtied(r_inner.dep2)
        resource_manager.finishedWith(r_inner)
        r_inner = resource_manager.getResource()
        dep2.dirtied(r_inner.dep2)
        resource_manager.finishedWith(r_inner)
        resource_manager.finishedWith(r_outer)
        self.assertEqual(1, dep1.makes)
        self.assertEqual(1, dep1.cleans)
        self.assertEqual(0, dep1.resets)
        self.assertEqual(1, dep2.makes)
        self.assertEqual(1, dep2.cleans)
        self.assertEqual(1, dep2.resets)
        self.assertEqual(1, resource_manager.makes)
        self.assertEqual(1, resource_manager.cleans)
        self.assertEqual(1, resource_manager.resets)

    def testDirtyingWhenUnused(self):
        resource_manager = MockResource()
        resource = resource_manager.getResource()
        resource_manager.finishedWith(resource)
        resource_manager.dirtied(resource)
        self.assertEqual(1, resource_manager.makes)
        resource = resource_manager.getResource()
        self.assertEqual(2, resource_manager.makes)

    def testFinishedActivityForResourceWithoutExtensions(self):
        result = ResultWithoutResourceExtensions()
        resource_manager = MockResource()
        r = resource_manager.getResource()
        resource_manager.finishedWith(r, result)

    def testFinishedActivityForResourceWithExtensions(self):
        result = ResultWithResourceExtensions()
        resource_manager = MockResource()
        r = resource_manager.getResource()
        expected = [('clean', 'start', resource_manager), ('clean', 'stop', resource_manager)]
        resource_manager.finishedWith(r, result)
        self.assertEqual(expected, result._calls)

    def testGetActivityForResourceWithoutExtensions(self):
        result = ResultWithoutResourceExtensions()
        resource_manager = MockResource()
        r = resource_manager.getResource(result)
        resource_manager.finishedWith(r)

    def testGetActivityForResourceWithExtensions(self):
        result = ResultWithResourceExtensions()
        resource_manager = MockResource()
        r = resource_manager.getResource(result)
        expected = [('make', 'start', resource_manager), ('make', 'stop', resource_manager)]
        resource_manager.finishedWith(r)
        self.assertEqual(expected, result._calls)

    def testResetActivityForResourceWithoutExtensions(self):
        result = ResultWithoutResourceExtensions()
        resource_manager = MockResource()
        resource_manager.getResource()
        r = resource_manager.getResource()
        resource_manager.dirtied(r)
        resource_manager.finishedWith(r)
        r = resource_manager.getResource(result)
        resource_manager.dirtied(r)
        resource_manager.finishedWith(r)
        resource_manager.finishedWith(resource_manager._currentResource)

    def testResetActivityForResourceWithExtensions(self):
        result = ResultWithResourceExtensions()
        resource_manager = MockResource()
        expected = [('reset', 'start', resource_manager), ('reset', 'stop', resource_manager)]
        resource_manager.getResource()
        r = resource_manager.getResource()
        resource_manager.dirtied(r)
        resource_manager.finishedWith(r)
        r = resource_manager.getResource(result)
        resource_manager.dirtied(r)
        resource_manager.finishedWith(r)
        resource_manager.finishedWith(resource_manager._currentResource)
        self.assertEqual(expected, result._calls)