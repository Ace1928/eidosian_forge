import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, List
class ExtendedNotifiers(HasTraits):

    def __init__(self, **traits):
        ok_listeners = [self.method_listener_0, self.method_listener_1, self.method_listener_2, self.method_listener_3, self.method_listener_4]
        for listener in ok_listeners:
            self._on_trait_change(listener, 'ok', dispatch='extended')
        fail_listeners = [self.failing_method_listener_0, self.failing_method_listener_1, self.failing_method_listener_2, self.failing_method_listener_3, self.failing_method_listener_4]
        for listener in fail_listeners:
            self._on_trait_change(listener, 'fail', dispatch='extended')
        super().__init__(**traits)
    ok = Float
    fail = Float
    rebind_calls_0 = List
    rebind_calls_1 = List
    rebind_calls_2 = List
    rebind_calls_3 = List
    rebind_calls_4 = List
    exceptions_from = List

    def method_listener_0(self):
        self.rebind_calls_0.append(True)

    def method_listener_1(self, new):
        self.rebind_calls_1.append(new)

    def method_listener_2(self, name, new):
        self.rebind_calls_2.append((name, new))

    def method_listener_3(self, obj, name, new):
        self.rebind_calls_3.append((obj, name, new))

    def method_listener_4(self, obj, name, old, new):
        self.rebind_calls_4.append((obj, name, old, new))

    def failing_method_listener_0(self):
        self.exceptions_from.append(0)
        raise Exception('error')

    def failing_method_listener_1(self, new):
        self.exceptions_from.append(1)
        raise Exception('error')

    def failing_method_listener_2(self, name, new):
        self.exceptions_from.append(2)
        raise Exception('error')

    def failing_method_listener_3(self, obj, name, new):
        self.exceptions_from.append(3)
        raise Exception('error')

    def failing_method_listener_4(self, obj, name, old, new):
        self.exceptions_from.append(4)
        raise Exception('error')