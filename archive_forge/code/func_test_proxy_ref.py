import unittest
def test_proxy_ref(self):
    from kivy.uix.behaviors.knspace import knspace
    from kivy.uix.widget import Widget
    w = Widget()
    knspace.widget1 = w
    self.assertIs(w.proxy_ref, knspace.widget1)
    knspace.widget1 = 55
    self.assertIs(55, knspace.widget1)