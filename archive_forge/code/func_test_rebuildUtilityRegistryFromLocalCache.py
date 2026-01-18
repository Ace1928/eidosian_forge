import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_rebuildUtilityRegistryFromLocalCache(self):

    class IFoo(Interface):
        """Does nothing"""

    class UtilityImplementingFoo:
        """Does nothing"""
    comps = self._makeOne()
    for i in range(30):
        comps.registerUtility(UtilityImplementingFoo(), IFoo, name='{}'.format(i))
    orig_generation = comps.utilities._generation
    orig_adapters = comps.utilities._adapters
    self.assertEqual(len(orig_adapters), 1)
    self.assertEqual(len(orig_adapters[0]), 1)
    self.assertEqual(len(orig_adapters[0][IFoo]), 30)
    orig_subscribers = comps.utilities._subscribers
    self.assertEqual(len(orig_subscribers), 1)
    self.assertEqual(len(orig_subscribers[0]), 1)
    self.assertEqual(len(orig_subscribers[0][IFoo]), 1)
    self.assertEqual(len(orig_subscribers[0][IFoo]['']), 30)
    new_adapters = comps.utilities._adapters = type(orig_adapters)()
    new_adapters.append({})
    d = new_adapters[0][IFoo] = {}
    for name in range(10):
        name = str(str(name))
        d[name] = orig_adapters[0][IFoo][name]
    self.assertNotEqual(orig_adapters, new_adapters)
    new_subscribers = comps.utilities._subscribers = type(orig_subscribers)()
    new_subscribers.append({})
    d = new_subscribers[0][IFoo] = {}
    d[''] = ()
    for name in range(5, 12):
        name = str(str(name))
        comp = orig_adapters[0][IFoo][name]
        d[''] += (comp,)
    rebuild_results_preflight = comps.rebuildUtilityRegistryFromLocalCache()
    self.assertEqual(comps.utilities._generation, orig_generation)
    self.assertEqual(rebuild_results_preflight, {'did_not_register': 10, 'needed_registered': 20, 'did_not_subscribe': 7, 'needed_subscribed': 23})
    rebuild_results = comps.rebuildUtilityRegistryFromLocalCache(rebuild=True)
    self.assertEqual(comps.utilities._generation, orig_generation + 1)
    self.assertEqual(rebuild_results_preflight, rebuild_results)
    self.assertEqual(new_adapters, orig_adapters)
    self.assertEqual(len(new_subscribers[0][IFoo]['']), len(orig_subscribers[0][IFoo]['']))
    for orig_subscriber in orig_subscribers[0][IFoo]['']:
        self.assertIn(orig_subscriber, new_subscribers[0][IFoo][''])
    preflight_after = comps.rebuildUtilityRegistryFromLocalCache()
    self.assertEqual(preflight_after, {'did_not_register': 30, 'needed_registered': 0, 'did_not_subscribe': 30, 'needed_subscribed': 0})
    rebuild_after = comps.rebuildUtilityRegistryFromLocalCache(rebuild=True)
    self.assertEqual(rebuild_after, preflight_after)
    self.assertEqual(comps.utilities._generation, orig_generation + 1)